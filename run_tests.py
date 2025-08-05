#!/usr/bin/env python3
"""
Test runner for Career Prediction System
Runs all unit tests and provides a summary report
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """Run all tests and return results"""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2, stream=StringIO())
    
    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def run_specific_test(test_module):
    """Run a specific test module"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def print_test_summary(result, duration):
    """Print a summary of test results"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Duration: {duration:.2f} seconds")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n❌ Some tests failed!")
        return False
    else:
        print("\n✅ All tests passed!")
        return True

def main():
    """Main function"""
    print("🧪 Career Prediction System - Test Suite")
    print("="*60)
    
    if len(sys.argv) > 1:
        # Run specific test
        test_module = sys.argv[1]
        print(f"Running specific test: {test_module}")
        result, duration = run_specific_test(test_module)
    else:
        # Run all tests
        print("Running all tests...")
        result, duration = run_all_tests()
    
    success = print_test_summary(result, duration)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 