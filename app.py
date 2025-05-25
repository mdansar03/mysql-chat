import os
from dotenv import load_dotenv
import json
import hashlib
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
# import openai
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import re
from openai import OpenAI
import sqlparse
from difflib import SequenceMatcher
import time
from functools import lru_cache
import Levenshtein
import secrets

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Generate a secure secret key for sessions
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'your-pinecone-api-key')
    # print(PINECONE_API_KEY, "PINECONE_API_KEY ============>")
    PINECONE_INDEX_NAME = 'mysql-index-query'
    PINECONE_ENVIRONMENT = 'us-east-1'
    
    # Query Configuration
    MAX_QUERY_RESULTS = int(os.getenv('MAX_QUERY_RESULTS', '10000'))  # Maximum results to return
    DEFAULT_QUERY_LIMIT = int(os.getenv('DEFAULT_QUERY_LIMIT', '1000'))  # Default limit if not specified

# Initialize OpenAI
try:
    if Config.OPENAI_API_KEY == 'your-openai-api-key':
        logger.warning("OpenAI API key not found in environment variables. Using default placeholder.")
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    client = None

# Initialize Pinecone
pc = Pinecone(api_key=Config.PINECONE_API_KEY)

# Connection manager to handle multiple database connections
class ConnectionManager:
    def __init__(self):
        self.connections = {}
    
    def create_connection_id(self, host: str, user: str, database: str) -> str:
        """Create a unique connection ID based on connection parameters"""
        return hashlib.md5(f"{host}:{user}:{database}".encode()).hexdigest()
    
    def add_connection(self, connection_id: str, connection_params: Dict[str, str]) -> Optional[mysql.connector.MySQLConnection]:
        """Create and store a new database connection"""
        try:
            connection = mysql.connector.connect(
                host=connection_params['host'],
                user=connection_params['user'],
                password=connection_params['password'],
                database=connection_params['database'],
                autocommit=True
            )
            self.connections[connection_id] = {
                'connection': connection,
                'params': connection_params,
                'created_at': datetime.now()
            }
            logger.info(f"Database connection established for ID: {connection_id}")
            return connection
        except Error as e:
            logger.error(f"Error creating connection: {str(e)}")
            raise e
    
    def get_connection(self, connection_id: str) -> Optional[mysql.connector.MySQLConnection]:
        """Get an existing connection or None if not found"""
        if connection_id in self.connections:
            conn_info = self.connections[connection_id]
            connection = conn_info['connection']
            
            # Check if connection is still alive
            try:
                if connection.is_connected():
                    return connection
                else:
                    # Try to reconnect
                    logger.info(f"Reconnecting for connection ID: {connection_id}")
                    new_connection = self.add_connection(connection_id, conn_info['params'])
                    return new_connection
            except:
                # Try to reconnect
                logger.info(f"Connection lost, reconnecting for ID: {connection_id}")
                try:
                    new_connection = self.add_connection(connection_id, conn_info['params'])
                    return new_connection
                except:
                    return None
        return None
    
    def remove_connection(self, connection_id: str):
        """Close and remove a connection"""
        if connection_id in self.connections:
            try:
                conn = self.connections[connection_id]['connection']
                if conn.is_connected():
                    conn.close()
                del self.connections[connection_id]
                logger.info(f"Connection removed for ID: {connection_id}")
            except Exception as e:
                logger.error(f"Error removing connection: {str(e)}")

# Initialize connection manager
connection_manager = ConnectionManager()

class MySQLSchemaProcessor:
    def __init__(self):
        self.index = None
        self.initialize_pinecone()
        self.common_sql_terms = {
            'select', 'from', 'where', 'group', 'by', 'order', 'having', 
            'join', 'inner', 'left', 'right', 'outer', 'on', 'and', 'or',
            'count', 'sum', 'avg', 'max', 'min', 'distinct', 'limit',
            'asc', 'desc', 'between', 'like', 'in', 'not', 'null', 'is'
        }
    
    def initialize_pinecone(self):
        """Initialize Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if Config.PINECONE_INDEX_NAME not in existing_indexes:
                pc.create_index(
                    name=Config.PINECONE_INDEX_NAME,
                    dimension=3072,  # text-embedding-3-large dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=Config.PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"Created new index: {Config.PINECONE_INDEX_NAME}")
            
            self.index = pc.Index(Config.PINECONE_INDEX_NAME)
            logger.info("Pinecone index initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def get_connection_from_session(self) -> Optional[mysql.connector.MySQLConnection]:
        """Get the current database connection from session"""
        connection_id = session.get('connection_id')
        if not connection_id:
            return None
        return connection_manager.get_connection(connection_id)
    
    def ensure_connection(self) -> bool:
        """Ensure database connection is available"""
        connection = self.get_connection_from_session()
        return connection is not None and connection.is_connected()
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table schema information"""
        connection = self.get_connection_from_session()
        if not connection:
            logger.error("No active database connection for schema retrieval")
            return {}
        
        try:
            cursor = connection.cursor(dictionary=True)
            database_name = session.get('database_name', 'database')
            
            # Get column information
            cursor.execute(f"""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE,
                    COLUMN_DEFAULT,
                    COLUMN_KEY,
                    EXTRA,
                    COLUMN_COMMENT
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{database_name}' 
                AND TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """)
            columns = cursor.fetchall()
            
            # Get foreign key relationships
            cursor.execute(f"""
                SELECT 
                    COLUMN_NAME,
                    REFERENCED_TABLE_NAME,
                    REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = '{database_name}'
                AND TABLE_NAME = '{table_name}'
                AND REFERENCED_TABLE_NAME IS NOT NULL
            """)
            foreign_keys = cursor.fetchall()
            
            # Get indexes
            cursor.execute(f"SHOW INDEX FROM {table_name}")
            indexes = cursor.fetchall()
            
            # Get sample data (first 5 rows)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()
            
            # Get table statistics
            cursor.execute(f"SELECT COUNT(*) as row_count FROM {table_name}")
            row_count = cursor.fetchone()['row_count']
            
            schema_info = {
                'table_name': table_name,
                'columns': columns,
                'foreign_keys': foreign_keys,
                'indexes': indexes,
                'sample_data': sample_data,
                'row_count': row_count
            }
            
            cursor.close()
            return schema_info
            
        except Error as e:
            logger.error(f"Error getting table schema: {str(e)}")
            return {}
    
    def create_embeddings(self, text: str) -> List[float]:
        """Create embeddings using OpenAI text-embedding-3-large"""
        try:
            if not client:
                logger.error("OpenAI client not initialized")
                return []
                
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return []
    
    def chunk_schema_data(self, schema_info: Dict[str, Any], database_name: str) -> List[Dict[str, Any]]:
        """Create meaningful chunks from schema information, including database name in IDs and metadata"""
        chunks = []
        table_name = schema_info['table_name']
        # Main table description chunk
        main_description = f"""
        Table: {table_name}
        Total Rows: {schema_info['row_count']}
        
        Columns:
        """
        for col in schema_info['columns']:
            main_description += f"""
        - {col['COLUMN_NAME']}: {col['DATA_TYPE']}
          Nullable: {col['IS_NULLABLE']}
          Key: {col['COLUMN_KEY'] or 'None'}
          Default: {col['COLUMN_DEFAULT'] or 'None'}
          Comment: {col['COLUMN_COMMENT'] or 'None'}
        """
        chunks.append({
            'id': f"{database_name}_{table_name}_main_schema",
            'content': main_description,
            'metadata': {
                'database_name': database_name,
                'table_name': table_name,
                'chunk_type': 'main_schema',
                'row_count': schema_info['row_count']
            }
        })
        # Foreign key relationships chunk
        if schema_info['foreign_keys']:
            fk_description = f"""
            Table: {table_name} - Foreign Key Relationships:
            """
            for fk in schema_info['foreign_keys']:
                fk_description += f"""
            - {fk['COLUMN_NAME']} references {fk['REFERENCED_TABLE_NAME']}.{fk['REFERENCED_COLUMN_NAME']}
            """
            chunks.append({
                'id': f"{database_name}_{table_name}_foreign_keys",
                'content': fk_description,
                'metadata': {
                    'database_name': database_name,
                    'table_name': table_name,
                    'chunk_type': 'foreign_keys'
                }
            })
        # Sample data chunk
        if schema_info['sample_data']:
            sample_description = f"""
            Table: {table_name} - Sample Data:
            """
            for i, row in enumerate(schema_info['sample_data'], 1):
                sample_description += f"\nRow {i}: {json.dumps(row, default=str)}"
            chunks.append({
                'id': f"{database_name}_{table_name}_sample_data",
                'content': sample_description,
                'metadata': {
                    'database_name': database_name,
                    'table_name': table_name,
                    'chunk_type': 'sample_data'
                }
            })
        return chunks
    
    def store_table_schema(self, table_name: str):
        """Process and store table schema in Pinecone, using database name in IDs and metadata"""
        logger.info(f"Processing table: {table_name}")
        # Get schema information
        schema_info = self.get_table_schema(table_name)
        if not schema_info:
            return False
        database_name = session.get('database_name', 'database')
        # Create chunks
        chunks = self.chunk_schema_data(schema_info, database_name)
        # Create embeddings and store in Pinecone
        vectors_to_upsert = []
        for chunk in chunks:
            try:
                # Create embedding
                embedding = self.create_embeddings(chunk['content'])
                if not embedding:
                    continue
                # Create vector
                vector = {
                    'id': chunk['id'],
                    'values': embedding,
                    'metadata': {
                        **chunk['metadata'],
                        'content': chunk['content'][:1000],  # Store truncated content
                        'timestamp': datetime.now().isoformat()
                    }
                }
                vectors_to_upsert.append(vector)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk['id']}: {str(e)}")
                continue
        # Upsert to Pinecone
        if vectors_to_upsert:
            try:
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Successfully stored {len(vectors_to_upsert)} vectors for table {table_name}")
                return True
            except Exception as e:
                logger.error(f"Error upserting vectors: {str(e)}")
                return False
        return False
    
    def query_similar_schemas(self, query: str, table_name: str = None, top_k: int = 5) -> List[Dict]:
        """Query similar schema information from Pinecone, filtering by database name and table name"""
        try:
            # Create query embedding
            query_embedding = self.create_embeddings(query)
            print(query_embedding, "query_embedding ============>")
            if not query_embedding:
                return []
            # Prepare filter
            filter_dict = {}
            database_name = session.get('database_name', 'database')
            if table_name:
                filter_dict['table_name'] = table_name
            if database_name:
                filter_dict['database_name'] = database_name
            # Query Pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            return response.matches
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return []
    
    def execute_mysql_query(self, query: str) -> Dict[str, Any]:
        """Execute MySQL query safely with result limit handling"""
        connection = self.get_connection_from_session()
        if not connection:
            return {'error': 'No active database connection'}
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Basic query validation
            query = query.strip()
            logger.info(f"Executing query: {query}")
            
            if not query.upper().startswith('SELECT'):
                return {'error': 'Only SELECT queries are allowed'}
            
            # Check if query already has a LIMIT clause
            query_upper = query.upper()
            has_limit = 'LIMIT' in query_upper
            
            # If no limit and query doesn't have one, add a default limit for safety
            if not has_limit and Config.DEFAULT_QUERY_LIMIT > 0:
                # First, get the total count
                count_query = f"SELECT COUNT(*) as total FROM ({query}) as subquery"
                try:
                    cursor.execute(count_query)
                    total_count = cursor.fetchone()['total']
                except Exception as count_error:
                    logger.warning(f"Could not get total count: {count_error}")
                    total_count = None
                
                # Add limit to the main query
                limited_query = f"{query} LIMIT {Config.MAX_QUERY_RESULTS}"
                logger.info(f"Executing limited query: {limited_query}")
                cursor.execute(limited_query)
            else:
                cursor.execute(query)
                total_count = None
            
            results = cursor.fetchall()
            actual_count = len(results)
            
            # If we hit the max limit, inform the user
            if actual_count == Config.MAX_QUERY_RESULTS:
                logger.warning(f"Query results limited to {Config.MAX_QUERY_RESULTS} rows")
            
            cursor.close()
            
            logger.info(f"Query executed successfully, returned {actual_count} rows")
            
            return {
                'success': True,
                'data': results,
                'row_count': actual_count,
                'total_count': total_count,
                'limited': actual_count == Config.MAX_QUERY_RESULTS
            }
            
        except Error as e:
            logger.error(f"MySQL query error: {str(e)}")
            return {'error': str(e)}
    
    def generate_sql_query(self, user_question: str, table_name: str) -> str:
        """Generate SQL query using OpenAI based on schema context with retry logic"""
        max_retries = 3
        retry_count = 0
        previous_error = None  # Store previous error for retry context
        
        # Get table context for spell correction
        table_columns = self.get_table_context(table_name)
        
        # Correct spelling in user question (but preserve email addresses and special patterns)
        corrected_question = self.correct_spelling_smart(user_question, table_columns)
        
        if corrected_question != user_question:
            logger.info(f"Corrected question: '{user_question}' -> '{corrected_question}'")
        
        while retry_count < max_retries:
            try:
                # Get relevant schema information
                schema_matches = self.query_similar_schemas(user_question, table_name, top_k=3)
                logger.info(f"Found {len(schema_matches)} schema matches")
                
                # Build context from schema matches
                context = ""
                for match in schema_matches:
                    context += f"\n{match.metadata.get('content', '')}\n"
                
                # Create prompt for SQL generation
                prompt = f"""
                You are a MySQL query generator. Based on the following database schema information and user question, generate a precise SELECT query.
                
                Database Schema Context:
                {context}
                
                User Question: {user_question}
                Target Table: {table_name}
                
                Available Columns: {', '.join(table_columns)}
                
                Important: If the user mentions specific values like email addresses, names, IDs, etc., use them EXACTLY as provided in WHERE clauses.
                For example: 
                - "details of moni@gmail.com" → WHERE email = 'moni@gmail.com'
                - "find user john" → WHERE name = 'john' OR name LIKE '%john%'
                
                Rules:
                1. Only generate SELECT queries
                2. Use proper MySQL syntax
                3. Include appropriate WHERE clauses if needed
                4. Use JOINs if the question involves multiple tables
                5. Include ORDER BY and LIMIT if appropriate
                6. Return only the SQL query, no explanations
                7. Ensure all column names are spelled correctly
                8. For email addresses or specific values mentioned in the question, use them exactly as provided
                9. When searching for a specific record by email or unique identifier, use exact match (=) not LIKE
                
                {f"Previous attempt failed with error: {previous_error}. Please fix the syntax and try again." if retry_count > 0 and previous_error else ""}
                
                SQL Query:
                """
                
                if not client:
                    logger.error("OpenAI client not initialized")
                    previous_error = "OpenAI client not initialized"
                    retry_count += 1
                    continue
                    
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0
                )
                
                sql_query = response.choices[0].message.content.strip()
                
                # Clean up the query
                sql_query = re.sub(r'```sql\s*', '', sql_query)
                sql_query = re.sub(r'```\s*$', '', sql_query)
                sql_query = sql_query.strip()
                
                logger.info(f"Generated SQL (attempt {retry_count + 1}): {sql_query}")
                
                # Validate the generated query
                validation_result = self.validate_sql_syntax(sql_query)
                
                if validation_result['valid']:
                    logger.info(f"Generated valid SQL query after {retry_count + 1} attempts")
                    return validation_result.get('formatted_query', sql_query)
                else:
                    previous_error = validation_result['error']
                    logger.warning(f"SQL validation failed (attempt {retry_count + 1}): {previous_error}")
                    retry_count += 1
                    
            except Exception as e:
                logger.error(f"Error generating SQL query (attempt {retry_count + 1}): {str(e)}")
                previous_error = str(e)
                retry_count += 1
        
        # If all retries failed, return empty string
        logger.error(f"Failed to generate valid SQL after {max_retries} attempts. Last error: {previous_error}")
        return ""
    
    def validate_sql_syntax(self, query: str) -> Dict[str, Any]:
        """Validate SQL syntax and structure"""
        try:
            logger.info(f"Validating SQL query: {query}")
            
            # Parse SQL using sqlparse
            parsed = sqlparse.parse(query)
            
            if not parsed:
                return {'valid': False, 'error': 'Empty or invalid query'}
            
            # Format the query for better validation
            formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
            
            # Basic validation checks
            query_upper = query.upper().strip()
            
            # Check if it's a SELECT query
            if not query_upper.startswith('SELECT'):
                return {'valid': False, 'error': 'Only SELECT queries are allowed'}
            
            # Check for dangerous keywords with word boundaries
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
            
            # Use regex to check for whole word matches only
            for keyword in dangerous_keywords:
                # Use word boundary regex to avoid matching keywords within column names
                pattern = r'\b' + keyword + r'\b'
                if re.search(pattern, query_upper):
                    return {'valid': False, 'error': f'Dangerous keyword {keyword} detected'}
            
            # Try to validate with MySQL syntax check (dry run) - but only if connection is available
            connection = self.get_connection_from_session()
            if connection and connection.is_connected():
                try:
                    cursor = connection.cursor()
                    # Use EXPLAIN to validate syntax without executing
                    explain_query = f"EXPLAIN {query}"
                    logger.info(f"Running EXPLAIN on query for validation")
                    cursor.execute(explain_query)
                    cursor.fetchall()
                    cursor.close()
                    logger.info("Query validation successful with EXPLAIN")
                    return {'valid': True, 'formatted_query': formatted_query}
                except Error as e:
                    # If EXPLAIN fails, it might be due to the query syntax or missing tables
                    # We'll let it pass basic validation and fail during execution with a clearer error
                    logger.warning(f"EXPLAIN validation failed (might be due to missing tables): {str(e)}")
                    # Still return as valid since basic syntax checks passed
                    return {'valid': True, 'formatted_query': formatted_query, 'warning': f'Could not validate with EXPLAIN: {str(e)}'}
            else:
                # No database connection available, but basic validation passed
                logger.info("Query passed basic validation (no DB connection for EXPLAIN)")
                return {'valid': True, 'formatted_query': formatted_query, 'warning': 'Validated without database connection'}
            
        except Exception as e:
            logger.error(f"Error during query validation: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def correct_spelling_smart(self, text: str, context_words: List[str] = None) -> str:
        """Smart spelling correction that preserves email addresses and special patterns"""
        # Preserve email addresses and other patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Find and store special patterns
        emails = re.findall(email_pattern, text)
        
        # Create placeholders for special patterns
        temp_text = text
        email_placeholders = {}
        for i, email in enumerate(emails):
            placeholder = f"__EMAIL_{i}__"
            email_placeholders[placeholder] = email
            temp_text = temp_text.replace(email, placeholder)
        
        # Now correct spelling on the modified text
        words = temp_text.lower().split()
        corrected_words = []
        
        # Combine SQL terms with table/column names for context
        if context_words:
            valid_terms = self.common_sql_terms.union(set(word.lower() for word in context_words))
        else:
            valid_terms = self.common_sql_terms
        
        for word in words:
            # Skip placeholders
            if word.startswith("__") and word.endswith("__"):
                corrected_words.append(word)
                continue
                
            # Skip if it's already a valid term or contains special characters
            if word in valid_terms or word.isdigit() or '@' in word or '.' in word:
                corrected_words.append(word)
                continue
            
            # For common query words, apply fuzzy matching
            best_match = word
            best_score = 0
            
            # Only check against SQL terms, not column names for generic words
            check_terms = self.common_sql_terms
            
            for valid_term in check_terms:
                similarity = 1 - (Levenshtein.distance(word, valid_term) / max(len(word), len(valid_term)))
                
                if similarity > best_score and similarity > 0.7:
                    best_score = similarity
                    best_match = valid_term
            
            corrected_words.append(best_match)
        
        # Reconstruct the corrected text
        corrected_text = ' '.join(corrected_words)
        
        # Restore special patterns
        for placeholder, original in email_placeholders.items():
            corrected_text = corrected_text.replace(placeholder.lower(), original)
        
        return corrected_text
    
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """Analyze query performance and provide optimization suggestions"""
        connection = self.get_connection_from_session()
        if not connection:
            return {'error': 'No database connection'}
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Get query execution plan
            cursor.execute(f"EXPLAIN {query}")
            execution_plan = cursor.fetchall()
            
            # Get extended execution plan
            cursor.execute(f"EXPLAIN FORMAT=JSON {query}")
            json_plan = cursor.fetchone()
            
            # Analyze the plan for optimization opportunities
            suggestions = []
            
            for row in execution_plan:
                # Check for full table scans
                if row.get('type') == 'ALL':
                    suggestions.append(f"Full table scan detected on table '{row.get('table')}'. Consider adding an index.")
                
                # Check for filesort
                if row.get('Extra') and 'filesort' in row.get('Extra'):
                    suggestions.append("Query uses filesort. Consider adding an index on ORDER BY columns.")
                
                # Check for temporary tables
                if row.get('Extra') and 'temporary' in row.get('Extra'):
                    suggestions.append("Query creates temporary tables. Consider optimizing GROUP BY or DISTINCT clauses.")
                
                # Check key usage
                if not row.get('key'):
                    suggestions.append(f"No index used for table '{row.get('table')}'. Consider adding appropriate indexes.")
            
            cursor.close()
            
            return {
                'execution_plan': execution_plan,
                'json_plan': json_plan,
                'suggestions': suggestions,
                'estimated_rows': sum(row.get('rows', 0) for row in execution_plan)
            }
            
        except Error as e:
            return {'error': str(e)}
    
    @lru_cache(maxsize=100)
    def get_table_context(self, table_name: str) -> List[str]:
        """Get table column names for spelling correction context"""
        # Clear cache when a new connection is made
        if hasattr(self, '_last_connection_id') and self._last_connection_id != session.get('connection_id'):
            self.get_table_context.cache_clear()
            self._last_connection_id = session.get('connection_id')
        
        connection = self.get_connection_from_session()
        if not connection:
            logger.warning(f"Could not get table context for {table_name} - no database connection")
            return []
        
        try:
            cursor = connection.cursor()
            cursor.execute(f"DESCRIBE {table_name}")
            columns = [row[0] for row in cursor.fetchall()]
            cursor.close()
            logger.info(f"Retrieved {len(columns)} columns for table {table_name}")
            return columns
        except Exception as e:
            logger.error(f"Error getting table context: {str(e)}")
            return []
    
    def enhance_response_with_ai(self, query_result: Dict[str, Any], user_question: str) -> str:
        """Use AI to create a natural language response from query results"""
        try:
            # Prepare context from results
            if not query_result.get('data'):
                return "No results found for your query."
            
            # Limit data for API call
            sample_data = query_result['data'][:10] if len(query_result['data']) > 10 else query_result['data']
            
            prompt = f"""
            Based on the following database query results, provide a clear, natural language summary that answers the user's question.
            
            User Question: {user_question}
            
            Query Results (showing {len(sample_data)} of {query_result['row_count']} total rows):
            {json.dumps(sample_data, indent=2, default=str)}
            
            Please provide a conversational response that:
            1. Directly answers the user's question
            2. Highlights key insights from the data
            3. Mentions any patterns or notable findings
            4. Suggests relevant follow-up questions if appropriate
            
            Format your response in a clear, readable way:
            - Use plain text without any HTML tags
            - Start with a brief summary
            - Use bullet points or numbered lists where appropriate
            - Keep paragraphs short and focused
            - If listing multiple items, format them clearly
            
            IMPORTANT:
            - Be concise but complete - finish all sentences and thoughts
            - Aim for 200-400 words for most responses
            - If you have follow-up questions, list them as bullet points at the end
            - DO NOT use HTML tags like <p>, <ul>, <li>, etc. Use plain text formatting only
            - Use line breaks for paragraphs and bullet points like "•" or "-" for lists
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Check if response might be truncated
            if response.choices[0].finish_reason == 'length':
                ai_response += "\n\n[Response was truncated due to length. Please ask a follow-up question for more details.]"
            
            # Also check for common truncation indicators
            elif ai_response and (
                ai_response.endswith(('...', '..', '.', ':')) == False and 
                not ai_response[-1] in '.!?"\')`]' and
                len(ai_response.split()) > 100
            ):
                # Likely truncated if doesn't end with proper punctuation and is long
                ai_response += "..."
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {str(e)}")
            return f"Found {query_result['row_count']} results for your query."

# Initialize processor
processor = MySQLSchemaProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/api/connect', methods=['POST'])
def connect_database():
    """Connect to a MySQL database with provided credentials"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['host', 'user', 'password', 'database']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Create connection parameters
        connection_params = {
            'host': data['host'],
            'user': data['user'],
            'password': data['password'],
            'database': data['database']
        }
        
        # Create connection ID
        connection_id = connection_manager.create_connection_id(
            data['host'], 
            data['user'], 
            data['database']
        )
        
        # Try to establish connection
        try:
            connection = connection_manager.add_connection(connection_id, connection_params)
            
            # Store connection info in session
            session['connection_id'] = connection_id
            session['database_name'] = data['database']
            
            # Clear any cached data from previous connections
            if hasattr(processor, 'get_table_context'):
                processor.get_table_context.cache_clear()
            
            return jsonify({
                'success': True,
                'message': f"Connected to database '{data['database']}' successfully",
                'database': data['database']
            })
            
        except Error as e:
            error_message = str(e)
            if 'Access denied' in error_message:
                return jsonify({'error': 'Access denied. Please check your username and password.'}), 401
            elif 'Unknown database' in error_message:
                return jsonify({'error': f"Database '{data['database']}' does not exist."}), 404
            elif "Can't connect" in error_message:
                return jsonify({'error': f"Cannot connect to MySQL server at '{data['host']}'. Please check the host and port."}), 503
            else:
                return jsonify({'error': f'Connection failed: {error_message}'}), 500
                
    except Exception as e:
        logger.error(f"Error in connect_database: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect_database():
    """Disconnect from the current database"""
    try:
        connection_id = session.get('connection_id')
        
        if connection_id:
            connection_manager.remove_connection(connection_id)
            session.pop('connection_id', None)
            session.pop('database_name', None)
            
            # Clear any cached data
            if hasattr(processor, 'get_table_context'):
                processor.get_table_context.cache_clear()
            
            return jsonify({
                'success': True,
                'message': 'Disconnected from database successfully'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No active connection to disconnect'
            })
            
    except Exception as e:
        logger.error(f"Error in disconnect_database: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/connection_status', methods=['GET'])
def connection_status():
    """Get current connection status"""
    try:
        connection_id = session.get('connection_id')
        database_name = session.get('database_name')
        
        if not connection_id:
            return jsonify({
                'connected': False,
                'message': 'No active database connection'
            })
        
        connection = connection_manager.get_connection(connection_id)
        if connection and connection.is_connected():
            return jsonify({
                'connected': True,
                'database': database_name,
                'message': f"Connected to '{database_name}'"
            })
        else:
            # Connection lost, clean up session
            session.pop('connection_id', None)
            session.pop('database_name', None)
            return jsonify({
                'connected': False,
                'message': 'Database connection lost'
            })
            
    except Exception as e:
        logger.error(f"Error in connection_status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_table', methods=['POST'])
def process_table():
    """Process and store table schema"""
    try:
        # Check if connected
        if not session.get('connection_id'):
            return jsonify({'error': 'No active database connection. Please connect to a database first.'}), 401
            
        data = request.get_json()
        table_name = data.get('table_name', '').strip()
        
        if not table_name:
            return jsonify({'error': 'Table name is required'}), 400
        
        success = processor.store_table_schema(table_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Table {table_name} processed and stored successfully'
            })
        else:
            return jsonify({'error': 'Failed to process table'}), 500
            
    except Exception as e:
        logger.error(f"Error in process_table: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle user queries with enhanced validation and AI responses"""
    try:
        # Check if connected
        if not session.get('connection_id'):
            return jsonify({'error': 'No active database connection. Please connect to a database first.'}), 401
            
        data = request.get_json()
        user_question = data.get('question', '').strip()
        table_name = data.get('table_name', '').strip()
        
        if not user_question or not table_name:
            return jsonify({'error': 'Question and table name are required'}), 400
        
        print(user_question, "user_question ============>")
        print(table_name, "table_name ============>")
        
        # Start timer for performance tracking
        start_time = time.time()
        
        # Generate SQL query with retry logic
        sql_query = processor.generate_sql_query(user_question, table_name)
        
        if not sql_query:
            return jsonify({
                'error': 'Failed to generate valid SQL query after multiple attempts',
                'suggestion': 'Please try rephrasing your question or check the table schema'
            }), 500
        
        # Execute query
        query_result = processor.execute_mysql_query(sql_query)
        
        if 'error' in query_result:
            return jsonify({
                'error': query_result['error'],
                'generated_query': sql_query,
                'suggestion': 'Query syntax is valid but execution failed. Check your data and permissions.'
            }), 400
        
        # Analyze query performance
        performance_analysis = processor.analyze_query_performance(sql_query)
        
        # Generate AI-enhanced response
        ai_response = processor.enhance_response_with_ai(query_result, user_question)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'question': user_question,
            'generated_query': sql_query,
            'results': query_result['data'],
            'row_count': query_result['row_count'],
            'total_count': query_result.get('total_count'),
            'limited': query_result.get('limited', False),
            'max_results': Config.MAX_QUERY_RESULTS,
            'ai_summary': ai_response,
            'performance': {
                'execution_time': f"{execution_time:.2f} seconds",
                'execution_plan': performance_analysis.get('execution_plan', []),
                'optimization_suggestions': performance_analysis.get('suggestions', []),
                'estimated_rows_examined': performance_analysis.get('estimated_rows', 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in handle_query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_tables', methods=['GET'])
def get_tables():
    """Get list of available tables"""
    try:
        # Check if connected
        if not session.get('connection_id'):
            return jsonify({'error': 'No active database connection. Please connect to a database first.'}), 401
            
        connection = processor.get_connection_from_session()
        if not connection:
            return jsonify({'error': 'Database connection lost'}), 500
        
        cursor = connection.cursor()
        database_name = session.get('database_name', 'database')
        cursor.execute(f"SHOW TABLES FROM `{database_name}`")
        tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        
        return jsonify({'tables': tables})
        
    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)