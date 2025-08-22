#!/usr/bin/env python3
"""
Test script for the AcademiTrend API endpoints
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, description):
    """Test a specific API endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Endpoint: {endpoint}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}{endpoint}")
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response Type: {type(data)}")
                print(f"Response Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                
                # Pretty print the response (truncated for readability)
                json_str = json.dumps(data, indent=2)
                if len(json_str) > 1000:
                    print("Response (first 1000 chars):")
                    print(json_str[:1000] + "...")
                else:
                    print("Response:")
                    print(json_str)
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Raw response: {response.text[:500]}...")
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection Error: Make sure the Flask app is running on localhost:5000")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Test all API endpoints"""
    print("AcademiTrend API Test Suite")
    print("Make sure the Flask app is running before executing this test")
    
    # Test endpoints
    endpoints = [
        ("/", "Root endpoint"),
        ("/api/hello", "Hello endpoint"),
        ("/api/forecast", "Path forecasting endpoint"),
        ("/api/path-forecast", "Path forecast endpoint (alias)"),
        ("/api/pathway-data", "Pathway enrollment data from enrollment_trend.csv"),
        ("/api/simple-course-enrollment-prediction", "Simple course enrollment prediction"),
        ("/api/course-enrollment-prediction", "Advanced course enrollment prediction"),
        ("/api/load-course-predictions", "Load existing course predictions"),
        ("/api/predictions", "All predictions combined")
    ]
    
    for endpoint, description in endpoints:
        test_endpoint(endpoint, description)
        time.sleep(1)  # Small delay between requests
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 