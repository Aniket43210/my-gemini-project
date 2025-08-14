import unittest
import sys

if __name__ == '__main__':
    # Add current directory to path
    sys.path.insert(0, '.')
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Load all tests
    suite = loader.discover('tests', pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=False)
    result = runner.run(suite)
    
    # Exit with appropriate status
    sys.exit(not result.wasSuccessful())
