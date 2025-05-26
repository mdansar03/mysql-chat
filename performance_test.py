#!/usr/bin/env python3
"""
Performance Test Script for MySQL AI Assistant
Tests query response times with different configurations
"""

import requests
import time
import json
import statistics

class PerformanceTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_query_performance(self, question, table_name, num_tests=3):
        """Test query performance multiple times and return statistics"""
        times = []
        
        for i in range(num_tests):
            start_time = time.time()
            
            try:
                response = self.session.post(
                    f"{self.base_url}/api/query",
                    json={
                        "question": question,
                        "table_name": table_name
                    },
                    timeout=60
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                times.append(response_time)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"Test {i+1}: {response_time:.2f}s - Success")
                        if 'performance' in data and 'timing_breakdown' in data['performance']:
                            breakdown = data['performance']['timing_breakdown']
                            print(f"  Breakdown: SQL={breakdown.get('sql_generation', 'N/A')}, "
                                  f"Query={breakdown.get('query_execution', 'N/A')}, "
                                  f"AI={breakdown.get('ai_response', 'N/A')}")
                    else:
                        print(f"Test {i+1}: {response_time:.2f}s - Failed: {data.get('error', 'Unknown error')}")
                else:
                    print(f"Test {i+1}: {response_time:.2f}s - HTTP Error: {response.status_code}")
                    
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                times.append(response_time)
                print(f"Test {i+1}: {response_time:.2f}s - Exception: {str(e)}")
            
            # Wait between tests
            if i < num_tests - 1:
                time.sleep(1)
        
        if times:
            return {
                'min': min(times),
                'max': max(times),
                'avg': statistics.mean(times),
                'median': statistics.median(times),
                'times': times
            }
        return None
    
    def set_performance_config(self, config):
        """Set performance configuration"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/performance_config",
                json=config
            )
            if response.status_code == 200:
                print(f"Performance config updated: {config}")
                return True
            else:
                print(f"Failed to update config: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error updating config: {str(e)}")
            return False
    
    def run_performance_comparison(self, question, table_name):
        """Run performance tests with different configurations"""
        
        print("=" * 60)
        print("MySQL AI Assistant Performance Test")
        print("=" * 60)
        
        # Test configurations
        configs = [
            {
                'name': 'Optimized (Fast)',
                'config': {
                    'use_faster_model': True,
                    'reduce_ai_context': True,
                    'enable_vector_search': False,
                    'enable_performance_analysis': False,
                    'enable_query_validation': False,
                    'max_retries': 1
                }
            },
            {
                'name': 'Standard (Balanced)',
                'config': {
                    'use_faster_model': True,
                    'reduce_ai_context': False,
                    'enable_vector_search': False,
                    'enable_performance_analysis': True,
                    'enable_query_validation': False,
                    'max_retries': 2
                }
            },
            {
                'name': 'Full Features (Slow)',
                'config': {
                    'use_faster_model': False,
                    'reduce_ai_context': False,
                    'enable_vector_search': True,
                    'enable_performance_analysis': True,
                    'enable_query_validation': True,
                    'max_retries': 3
                }
            }
        ]
        
        results = {}
        
        for config_info in configs:
            print(f"\nTesting {config_info['name']} configuration...")
            print("-" * 40)
            
            # Set configuration
            if self.set_performance_config(config_info['config']):
                time.sleep(2)  # Wait for config to take effect
                
                # Run tests
                stats = self.test_query_performance(question, table_name, num_tests=3)
                
                if stats:
                    results[config_info['name']] = stats
                    print(f"\nResults for {config_info['name']}:")
                    print(f"  Average: {stats['avg']:.2f}s")
                    print(f"  Min: {stats['min']:.2f}s")
                    print(f"  Max: {stats['max']:.2f}s")
                    print(f"  Median: {stats['median']:.2f}s")
                else:
                    print(f"No valid results for {config_info['name']}")
        
        # Summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for name, stats in results.items():
            print(f"{name:20} | Avg: {stats['avg']:6.2f}s | Min: {stats['min']:6.2f}s | Max: {stats['max']:6.2f}s")
        
        if results:
            fastest = min(results.items(), key=lambda x: x[1]['avg'])
            slowest = max(results.items(), key=lambda x: x[1]['avg'])
            
            improvement = ((slowest[1]['avg'] - fastest[1]['avg']) / slowest[1]['avg']) * 100
            
            print(f"\nFastest: {fastest[0]} ({fastest[1]['avg']:.2f}s)")
            print(f"Slowest: {slowest[0]} ({slowest[1]['avg']:.2f}s)")
            print(f"Performance improvement: {improvement:.1f}%")
            
            if fastest[1]['avg'] <= 5.0:
                print("✅ Target of ≤5 seconds achieved!")
            else:
                print("❌ Target of ≤5 seconds not achieved")

def main():
    # Configuration
    BASE_URL = "http://localhost:5000"
    TEST_QUESTION = "show me user details"  # Modify this for your test
    TEST_TABLE = "users"  # Modify this for your test table
    
    print("Starting performance test...")
    print(f"Base URL: {BASE_URL}")
    print(f"Test Question: '{TEST_QUESTION}'")
    print(f"Test Table: '{TEST_TABLE}'")
    
    tester = PerformanceTester(BASE_URL)
    tester.run_performance_comparison(TEST_QUESTION, TEST_TABLE)

if __name__ == "__main__":
    main() 