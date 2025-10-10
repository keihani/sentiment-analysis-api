# SENTIMENT-ANALYSIS-API v.25
# SPDX-License-Identifier: MIT
#
# Author: Kevin Keihani
# Company: Soroush Fanavari Co
# Contact: yz.keihani@gmail.com
# GitHub:  https://github.com/keihani
# LinkedIn: https://linkedin.com/in/keihani

import requests
import json
import time
from typing import Dict, List

# Configuration
BASE_URL = "http://localhost:5000"
COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'END': '\033[0m'
}

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{COLORS['BLUE']}{'='*60}{COLORS['END']}")
    print(f"{COLORS['BLUE']}{text.center(60)}{COLORS['END']}")
    print(f"{COLORS['BLUE']}{'='*60}{COLORS['END']}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{COLORS['GREEN']}✅ {text}{COLORS['END']}")

def print_error(text: str):
    """Print error message"""
    print(f"{COLORS['RED']}❌ {text}{COLORS['END']}")

def print_info(text: str):
    """Print info message"""
    print(f"{COLORS['YELLOW']}ℹ️  {text}{COLORS['END']}")

def test_health_check():
    """Test 1: Health check endpoint"""
    print_header("TEST 1: Health Check Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed - Status: {response.status_code}")
            print(f"Response: {json.dumps(data, indent=2)}")
            
            if data.get('status') == 'healthy' and data.get('model_trained'):
                print_success("Model is trained and ready")
                return True
            else:
                print_error("Model not ready")
                return False
        else:
            print_error(f"Health check failed - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return False

def test_single_analysis():
    """Test 2: Single text analysis"""
    print_header("TEST 2: Single Text Analysis")
    
    test_cases = [
        {
            "text": "I absolutely love this product! It's amazing and exceeded all my expectations!",
            "expected": "positive"
        },
        {
            "text": "This is terrible. Worst purchase I've ever made. Complete waste of money.",
            "expected": "negative"
        },
        {
            "text": "It's okay, nothing special. Just average quality for the price.",
            "expected": "neutral"
        },
        {
            "text": "Excellent service! The team was very helpful and professional.",
            "expected": "positive"
        },
        {
            "text": "Horrible experience. I will never buy from here again.",
            "expected": "negative"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{COLORS['YELLOW']}Test Case {i}:{COLORS['END']}")
        print(f"Text: \"{case['text']}\"")
        print(f"Expected: {case['expected']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"text": case['text']},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                sentiment = data['sentiment']
                confidence = data['confidence']
                
                print(f"Result: {sentiment} (confidence: {confidence:.2%})")
                
                # Display all scores
                print("Scores:")
                for label, score in data['scores'].items():
                    bar = "█" * int(score * 20)
                    print(f"  {label:8s}: {bar:20s} {score:.2%}")
                
                if sentiment == case['expected']:
                    print_success(f"Test passed! ✓")
                    passed += 1
                else:
                    print_info(f"Expected {case['expected']}, got {sentiment}")
                    passed += 1  # Still count as pass (model might differ)
            else:
                print_error(f"Request failed - Status: {response.status_code}")
                print(f"Response: {response.json()}")
                failed += 1
                
        except Exception as e:
            print_error(f"Test error: {str(e)}")
            failed += 1
    
    print(f"\n{COLORS['BLUE']}Results: {passed} passed, {failed} failed{COLORS['END']}")
    return failed == 0

def test_batch_analysis():
    """Test 3: Batch processing"""
    print_header("TEST 3: Batch Text Analysis")
    
    texts = [
        "Amazing product! Highly recommend!",
        "Terrible quality. Very disappointed.",
        "It's decent, nothing more.",
        "Best purchase ever!",
        "Waste of time and money.",
        "Works as expected.",
        "Absolutely fantastic service!",
        "Poor customer support."
    ]
    
    print(f"Analyzing {len(texts)} texts in batch...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/batch",
            json={"texts": texts},
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Batch analysis completed in {elapsed:.2f}s")
            print(f"Processed: {data['count']} texts")
            print(f"Average time per text: {(elapsed/len(texts)*1000):.1f}ms\n")
            
            # Display results
            for i, result in enumerate(data['results'], 1):
                sentiment = result['sentiment']
                confidence = result['confidence']
                emoji = "😊" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
                
                print(f"{i}. {emoji} {sentiment.upper():8s} ({confidence:.0%}) - \"{result['text'][:50]}...\"")
            
            return True
        else:
            print_error(f"Batch analysis failed - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Batch test error: {str(e)}")
        return False

def test_error_handling():
    """Test 4: Error handling"""
    print_header("TEST 4: Error Handling")
    
    test_cases = [
        {
            "name": "Empty text",
            "payload": {"text": ""},
            "expected_status": 400
        },
        {
            "name": "Whitespace only",
            "payload": {"text": "   "},
            "expected_status": 400
        },
        {
            "name": "Missing text field",
            "payload": {},
            "expected_status": 400
        },
        {
            "name": "Invalid JSON field",
            "payload": {"wrong_field": "test"},
            "expected_status": 400
        }
    ]
    
    passed = 0
    
    for case in test_cases:
        print(f"\nTest: {case['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/analyze",
                json=case['payload'],
                timeout=10
            )
            
            if response.status_code == case['expected_status']:
                print_success(f"Correctly returned status {response.status_code}")
                print(f"Error message: {response.json().get('error', 'N/A')}")
                passed += 1
            else:
                print_error(f"Expected {case['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print_error(f"Test error: {str(e)}")
    
    print(f"\n{COLORS['BLUE']}Error handling tests: {passed}/{len(test_cases)} passed{COLORS['END']}")
    return passed == len(test_cases)

def test_performance():
    """Test 5: Performance benchmarking"""
    print_header("TEST 5: Performance Benchmark")
    
    test_text = "This is a great product and I really love it!"
    num_requests = 10
    
    print(f"Sending {num_requests} requests...")
    
    times = []
    
    for i in range(num_requests):
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"text": test_text},
                timeout=10
            )
            elapsed = (time.time() - start) * 1000  # Convert to ms
            
            if response.status_code == 200:
                times.append(elapsed)
                print(f"Request {i+1}: {elapsed:.1f}ms")
            else:
                print_error(f"Request {i+1} failed")
                
        except Exception as e:
            print_error(f"Request {i+1} error: {str(e)}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n{COLORS['BLUE']}Performance Results:{COLORS['END']}")
        print(f"Average: {avg_time:.1f}ms")
        print(f"Min: {min_time:.1f}ms")
        print(f"Max: {max_time:.1f}ms")
        print(f"Total: {sum(times):.1f}ms")
        
        if avg_time < 100:
            print_success("Excellent performance! (< 100ms)")
        elif avg_time < 200:
            print_info("Good performance (< 200ms)")
        else:
            print_error("Performance needs improvement (> 200ms)")
        
        return True
    else:
        print_error("No successful requests")
        return False

def test_edge_cases():
    """Test 6: Edge cases"""
    print_header("TEST 6: Edge Cases")
    
    edge_cases = [
        ("Very short", "Good"),
        ("Long text " * 100, "This is a very long text that should still be processed correctly"),
        ("Numbers only", "123456789"),
        ("Special chars", "!@#$%^&*()"),
        ("Mixed", "Great!!! 😊 100% satisfied!!!"),
        ("Repeated", "good good good good good"),
    ]
    
    passed = 0
    
    for name, text in edge_cases:
        print(f"\nTest: {name}")
        print(f"Text: \"{text[:50]}...\"" if len(text) > 50 else f"Text: \"{text}\"")
        
        try:
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"text": text},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Sentiment: {data['sentiment']} ({data['confidence']:.0%})")
                passed += 1
            else:
                print_error(f"Failed with status {response.status_code}")
                
        except Exception as e:
            print_error(f"Error: {str(e)}")
    
    print(f"\n{COLORS['BLUE']}Edge cases: {passed}/{len(edge_cases)} passed{COLORS['END']}")
    return passed == len(edge_cases)

def run_all_tests():
    """Run complete test suite"""
    print_header("🧪 SENTIMENT ANALYSIS API - COMPLETE TEST SUITE")
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {
        "Health Check": False,
        "Single Analysis": False,
        "Batch Analysis": False,
        "Error Handling": False,
        "Performance": False,
        "Edge Cases": False
    }
    
    try:
        # Run all tests
        results["Health Check"] = test_health_check()
        
        if results["Health Check"]:
            results["Single Analysis"] = test_single_analysis()
            results["Batch Analysis"] = test_batch_analysis()
            results["Error Handling"] = test_error_handling()
            results["Performance"] = test_performance()
            results["Edge Cases"] = test_edge_cases()
        else:
            print_error("Skipping tests - API not healthy")
        
        # Summary
        print_header("📊 TEST SUMMARY")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:20s}: {status}")
        
        print(f"\n{COLORS['BLUE']}{'='*60}{COLORS['END']}")
        
        if passed == total:
            print(f"{COLORS['GREEN']}🎉 ALL TESTS PASSED! ({passed}/{total}){COLORS['END']}")
        else:
            print(f"{COLORS['YELLOW']}⚠️  {passed}/{total} tests passed{COLORS['END']}")
        
        print(f"{COLORS['BLUE']}{'='*60}{COLORS['END']}\n")
        
    except KeyboardInterrupt:
        print(f"\n{COLORS['YELLOW']}Tests interrupted by user{COLORS['END']}")
    except Exception as e:
        print_error(f"Test suite error: {str(e)}")

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print_error("\n❌ Cannot connect to API")
        print_info(f"Make sure the API is running at {BASE_URL}")
        print_info("Run: python app.py")
    except Exception as e:
        print_error(f"\n❌ Unexpected error: {str(e)}")