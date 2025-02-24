import requests
import json
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api():
    logger.info("Starting API tests...")
    # Add delay to ensure server is ready
    time.sleep(2)
    base_url = "http://localhost:5000/api/analyze"
    
    test_cases = [
        {
            "name": "Empty text",
            "payload": {"text": ""},
            "expected_status": 400
        },
        {
            "name": "Text too short",
            "payload": {"text": "hi"},
            "expected_status": 400
        },
        {
            "name": "Valid text",
            "payload": {"text": "This is a genuine review about a product that I really enjoyed using. The quality is great and the price is reasonable."},
            "expected_status": 200
        },
        {
            "name": "Missing text field",
            "payload": {},
            "expected_status": 400
        },
        {
            "name": "Invalid JSON",
            "payload": None,
            "expected_status": 400
        }
    ]
    
    headers = {"Content-Type": "application/json"}
    success_count = 0
    total_tests = len(test_cases)
    
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        
        # Add delay between requests to avoid rate limiting
        time.sleep(3)  # Increased delay
        
        try:
            if test_case["payload"] is None:
                # Test invalid JSON
                response = requests.post(base_url, data="invalid json", headers=headers)
            else:
                response = requests.post(base_url, json=test_case["payload"], headers=headers)
            
            logger.info(f"Status Code: {response.status_code}")
            
            try:
                response_data = response.json()
                logger.info(f"Response: {response_data}")
                
                # Log detailed error message if available
                if response.status_code != 200 and 'details' in response_data:
                    logger.info(f"Error Details: {response_data['details']}")
                
            except json.JSONDecodeError:
                logger.info(f"Raw Response: {response.text}")
            
            if response.status_code == 429:
                logger.warning("Rate limit hit, waiting longer before next request...")
                time.sleep(10)  # Increased wait time for rate limit
                continue
                
            if response.status_code == test_case["expected_status"]:
                logger.info("Test passed! ✓")
                success_count += 1
            else:
                logger.error(f"Test failed: Expected status {test_case['expected_status']}, got {response.status_code} ✗")
                
        except requests.exceptions.ConnectionError:
            logger.error("Connection failed. Is the server running?")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
    
    # Print summary
    logger.info(f"\nTest Summary:")
    logger.info(f"Passed: {success_count}/{total_tests} tests")
    logger.info(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
            
if __name__ == "__main__":
    test_api() 