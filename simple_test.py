#!/usr/bin/env python3
"""
Simple test runner to identify issues
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.data_generator import SyntheticDataGenerator
        print("✓ data_generator imported successfully")
    except Exception as e:
        print(f"✗ data_generator import failed: {e}")
    
    try:
        from src.feature_engineering import create_academic_features
        print("✓ feature_engineering imported successfully")
    except Exception as e:
        print(f"✗ feature_engineering import failed: {e}")
    
    try:
        from src.ensemble_models import create_ensemble_model
        print("✓ ensemble_models imported successfully")
    except Exception as e:
        print(f"✗ ensemble_models import failed: {e}")
    
    try:
        from config.hobby_taxonomy import CAREER_HIERARCHY
        print("✓ hobby_taxonomy imported successfully")
    except Exception as e:
        print(f"✗ hobby_taxonomy import failed: {e}")
    
    try:
        from src.enhanced_cross_validation import EnhancedCrossValidation
        print("✓ enhanced_cross_validation imported successfully")
    except Exception as e:
        print(f"✗ enhanced_cross_validation import failed: {e}")

def test_simple_functionality():
    """Test simple functionality without heavy computations"""
    print("\nTesting simple functionality...")
    
    try:
        from config.hobby_taxonomy import get_career_field
        result = get_career_field('software_engineer')
        print(f"✓ get_career_field works: {result}")
    except Exception as e:
        print(f"✗ get_career_field failed: {e}")
    
    try:
        from src.feature_engineering import create_academic_features
        test_data = {'mathematics': 0.85, 'science': 0.80}
        result = create_academic_features(test_data)
        print(f"✓ create_academic_features works: {len(result)} features created")
    except Exception as e:
        print(f"✗ create_academic_features failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_simple_functionality()
    print("\nBasic tests completed.") 