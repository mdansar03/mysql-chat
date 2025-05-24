# MySQL AI Query Assistant - Enhanced Version

An intelligent MySQL query assistant that converts natural language questions into SQL queries with advanced features including spell checking, query validation, performance analysis, AI-enhanced responses, and efficient handling of large datasets.

## 🚀 New Features

### 1. **Intelligent Query Validation**
- Validates SQL syntax before execution
- Retry logic with up to 3 attempts to generate valid SQL
- Protection against dangerous SQL operations (DROP, DELETE, etc.)
- Automatic query formatting and optimization

### 2. **Spell Checking & Fuzzy Matching**
- Automatically corrects spelling mistakes in user queries
- Context-aware correction using table and column names
- Fuzzy matching with Levenshtein distance algorithm
- 70% similarity threshold for accurate corrections

### 3. **Performance Analysis**
- Query execution plan analysis
- Identifies performance bottlenecks:
  - Full table scans
  - Missing indexes
  - Temporary table usage
  - Filesort operations
- Provides optimization suggestions
- Execution time tracking

### 4. **AI-Enhanced Responses**
- Natural language summaries of query results
- Key insights and pattern identification
- Trend analysis from data
- Follow-up query suggestions
- User-friendly explanations of complex results

### 5. **Efficient Large Dataset Handling**
- **Pagination**: Load data in chunks with "Load More" functionality
- **View Modes**: Toggle between Card View and Table View
- **Export Options**: Export all data to CSV or JSON formats
- **Configurable Page Size**: Choose from 25, 50, 100, 200, or 500 records per page
- **Result Limits**: Automatic limiting of very large result sets with warnings
- **Performance Optimization**: Table view with sticky headers for efficient scrolling
- **Show All Option**: Display all records at once when needed

## 📋 Prerequisites

- Python 3.8+
- MySQL Server
- OpenAI API Key
- Pinecone API Key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mysql-query-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key

# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=your-database

# Query Limits (Optional)
MAX_QUERY_RESULTS=10000    # Maximum results to return (default: 10000)
DEFAULT_QUERY_LIMIT=1000   # Default limit if not specified (default: 1000)
```

## 🚀 Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Follow these steps:
   - Enter or select a table name
   - Click "Process Table" to analyze the schema
   - Ask questions in natural language

## 💡 Example Queries

The system handles various types of queries, including those with spelling mistakes:

### Basic Queries:
- "Show me all records from the table"
- "Find customers with orders over $100"
- "What are the top 10 products by sales?"

### Queries with Spelling Mistakes (Automatically Corrected):
- "selct all recrods from tabel" → "select all records from table"
- "find custmers with ordes over 100" → "find customers with orders over 100"
- "wat are the top prodcts" → "what are the top products"

## 🏗️ Architecture

### Query Processing Flow:
1. **User Input** → Spell checking and correction
2. **Context Retrieval** → Fetch relevant schema from Pinecone
3. **SQL Generation** → OpenAI generates SQL with retry logic
4. **Validation** → Syntax validation and safety checks
5. **Execution** → Query execution with performance analysis
6. **AI Enhancement** → Natural language response generation

### Key Components:

#### `MySQLSchemaProcessor`
- **`validate_sql_syntax()`**: Validates SQL syntax and checks for dangerous operations
- **`correct_spelling()`**: Corrects spelling mistakes using Levenshtein distance
- **`analyze_query_performance()`**: Analyzes execution plans and suggests optimizations
- **`enhance_response_with_ai()`**: Generates natural language summaries of results
- **`generate_sql_query()`**: Creates SQL with retry logic and validation

## 🔧 Configuration

### Performance Tuning:
- Adjust `max_retries` in `generate_sql_query()` (default: 3)
- Modify similarity threshold in `correct_spelling()` (default: 0.7)
- Change AI model in `client.chat.completions.create()` (default: gpt-4)

### Security Settings:
- Dangerous keywords list in `validate_sql_syntax()`
- Query type restrictions (SELECT only by default)

## 📊 Performance Optimization Tips

The system automatically analyzes queries and provides suggestions such as:
- Adding indexes for columns used in WHERE clauses
- Optimizing JOIN operations
- Avoiding full table scans
- Reducing filesort operations

## 🔒 Security Features

- SQL injection prevention
- Restricted to SELECT queries only
- Validation against dangerous operations
- Parameterized query execution

## 🐛 Troubleshooting

### Common Issues:

1. **"Failed to generate valid SQL after multiple attempts"**
   - Check if the table is properly processed
   - Verify column names in your question
   - Try rephrasing the question

2. **Spelling correction not working**
   - Ensure python-Levenshtein is installed
   - Check if table context is loaded

3. **Performance analysis missing**
   - Verify MySQL user has EXPLAIN privileges
   - Check database connection

## 📝 API Endpoints

- `POST /api/process_table` - Process and store table schema
- `POST /api/query` - Execute natural language query
- `GET /api/get_tables` - List available tables

### Example API Response:
```json
{
  "success": true,
  "question": "Show top customers",
  "generated_query": "SELECT * FROM customers ORDER BY total_purchases DESC LIMIT 10",
  "results": [...],
  "row_count": 10,
  "total_count": 1500,
  "limited": false,
  "max_results": 10000,
  "ai_summary": "The top 10 customers by total purchases are...",
  "performance": {
    "execution_time": "0.12 seconds",
    "optimization_suggestions": ["Consider adding index on total_purchases"],
    "estimated_rows_examined": 1000
  }
}
```

## 🔄 Handling Large Datasets

The application provides several strategies for working with large datasets:

1. **Automatic Result Limiting**: Queries returning more than MAX_QUERY_RESULTS rows are automatically limited
2. **Pagination Controls**: Load data incrementally with customizable page sizes
3. **Export Functionality**: Export all results to CSV or JSON for external analysis
4. **View Modes**: Switch between Card View (detailed) and Table View (compact)
5. **Performance Warnings**: Clear notifications when results are limited

### Best Practices for Large Datasets:
- Use specific WHERE clauses to filter data
- Add LIMIT clauses to your queries
- Use the Table View for scanning many records quickly
- Export to CSV/JSON for data analysis in external tools
- Adjust page size based on your needs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
