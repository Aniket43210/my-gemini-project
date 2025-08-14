"""
Unit tests for feature_engineering.py
Tests the feature engineering functions
"""

import unittest
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')

from src.feature_engineering import (
    create_academic_features,
    create_personality_features,
    create_hobby_features,
    create_derived_features,
    create_advanced_features,
    create_orientation_features,
    create_specialization_indices,
    create_all_features
)


class TestFeatureEngineering(unittest.TestCase):
    """Unit tests for feature engineering functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        self.sample_data = {
            'academic_grades': {
                'mathematics': 0.85,
                'science': 0.80,
                'english': 0.75,
                'social_science': 0.60,
                'second_language': 0.65
            },
            'personality': {
                'openness': 0.85,
                'conscientiousness': 0.75,
                'extraversion': 0.45,
                'agreeableness': 0.65,
                'neuroticism': 0.35
            },
            'hobbies': {
                'programming': {'intensity': 0.9, 'proficiency': 0.8, 'years': 4},
                'reading': {'intensity': 0.7, 'proficiency': 0.6, 'years': 2},
                'gaming': {'intensity': 0.5, 'proficiency': 0.4, 'years': 1}
            }
        }
    
    def test_create_academic_features(self):
        """Test academic feature creation"""
        features = create_academic_features(self.sample_data['academic_grades'])
        
        # Check required features exist
        required_features = ['mathematics', 'science', 'english', 'social_science', 'second_language']
        for feature in required_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
            self.assertGreaterEqual(features[feature], 0.0)
            self.assertLessEqual(features[feature], 1.0)
    
    def test_create_personality_features(self):
        """Test personality feature creation"""
        features = create_personality_features(self.sample_data['personality'])
        
        # Check required features exist
        required_features = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for feature in required_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
            self.assertGreaterEqual(features[feature], 0.0)
            self.assertLessEqual(features[feature], 1.0)
    
    def test_create_hobby_features(self):
        """Test hobby feature creation"""
        features = create_hobby_features(self.sample_data['hobbies'])
        
        # Check basic hobby features
        self.assertIn('hobby_count', features)
        self.assertIn('avg_intensity', features)
        self.assertIn('avg_proficiency', features)
        self.assertIn('avg_years', features)
        self.assertIn('max_intensity', features)
        self.assertIn('max_proficiency', features)
        self.assertIn('max_years', features)
        
        # Check binary hobby features
        self.assertIn('has_programming', features)
        self.assertIn('has_reading', features)
        self.assertIn('has_gaming', features)
        
        # Check data types and ranges
        self.assertIsInstance(features['hobby_count'], int)
        self.assertGreaterEqual(features['hobby_count'], 0)
        
        for feature in ['avg_intensity', 'avg_proficiency', 'max_intensity', 'max_proficiency']:
            self.assertIsInstance(features[feature], float)
            self.assertGreaterEqual(features[feature], 0.0)
            self.assertLessEqual(features[feature], 1.0)
        
        for feature in ['avg_years', 'max_years']:
            self.assertIsInstance(features[feature], (int, float))
            self.assertGreaterEqual(features[feature], 0)
        
        for feature in ['has_programming', 'has_reading', 'has_gaming']:
            self.assertIsInstance(features[feature], bool)
    
    def test_create_derived_features(self):
        """Test derived feature creation"""
        features = create_derived_features(self.sample_data['academic_grades'])
        
        # Check derived features
        self.assertIn('academic_peak', features)
        self.assertIn('academic_consistency', features)
        self.assertIn('stem_score', features)
        self.assertIn('humanities_score', features)
        self.assertIn('stem_vs_humanities', features)
        self.assertIn('overall_academic', features)
        
        # Check data types and ranges
        for feature in ['academic_peak', 'academic_consistency', 'stem_score', 
                       'humanities_score', 'stem_vs_humanities', 'overall_academic']:
            self.assertIsInstance(features[feature], float)
            self.assertGreaterEqual(features[feature], 0.0)
            self.assertLessEqual(features[feature], 1.0)
    
    def test_create_advanced_features(self):
        """Test advanced feature creation"""
        features = create_advanced_features(self.sample_data['personality'])
        
        # Check advanced personality features
        self.assertIn('leadership_potential', features)
        self.assertIn('analytical_disposition', features)
        self.assertIn('social_orientation', features)
        self.assertIn('emotional_stability', features)
        self.assertIn('adaptability', features)
        self.assertIn('detail_orientation', features)
        self.assertIn('creativity_index', features)
        self.assertIn('stress_tolerance', features)
        
        # Check data types and ranges
        for feature in features.values():
            self.assertIsInstance(feature, float)
            self.assertGreaterEqual(feature, 0.0)
            self.assertLessEqual(feature, 1.0)
    
    def test_create_orientation_features(self):
        """Test orientation feature creation"""
        features = create_orientation_features(
            self.sample_data['academic_grades'],
            self.sample_data['personality'],
            self.sample_data['hobbies']
        )
        
        # Check orientation features
        self.assertIn('technical_orientation', features)
        self.assertIn('creative_orientation', features)
        self.assertIn('social_orientation', features)
        self.assertIn('research_orientation', features)
        
        # Check data types and ranges
        for feature in features.values():
            self.assertIsInstance(feature, float)
            self.assertGreaterEqual(feature, 0.0)
            self.assertLessEqual(feature, 1.0)
    
    def test_create_specialization_indices(self):
        """Test specialization index creation"""
        features = create_specialization_indices(
            self.sample_data['academic_grades'],
            self.sample_data['personality'],
            self.sample_data['hobbies']
        )
        
        # Check specialization indices
        self.assertIn('hobby_specialization', features)
        self.assertIn('academic_specialization', features)
        self.assertIn('personality_specialization', features)
        
        # Check data types and ranges
        for feature in features.values():
            self.assertIsInstance(feature, float)
            self.assertGreaterEqual(feature, 0.0)
            self.assertLessEqual(feature, 1.0)
    
    def test_create_all_features(self):
        """Test complete feature creation"""
        features = create_all_features(self.sample_data)
        
        # Check that all feature categories are present
        self.assertIn('mathematics', features)  # Academic
        self.assertIn('openness', features)     # Personality
        self.assertIn('hobby_count', features)  # Hobby
        self.assertIn('academic_peak', features)  # Derived
        self.assertIn('leadership_potential', features)  # Advanced
        self.assertIn('technical_orientation', features)  # Orientation
        self.assertIn('hobby_specialization', features)  # Specialization
        
        # Check total number of features (should be around 51)
        self.assertGreater(len(features), 40)
        self.assertLess(len(features), 60)
        
        # Check all features are numeric
        for feature_name, feature_value in features.items():
            self.assertIsInstance(feature_value, (int, float, bool))
            if isinstance(feature_value, (int, float)) and not isinstance(feature_value, bool):
                self.assertGreaterEqual(feature_value, 0.0)
                self.assertLessEqual(feature_value, 1.0)
    
    def test_empty_hobbies_handling(self):
        """Test handling of empty hobbies"""
        data_with_empty_hobbies = self.sample_data.copy()
        data_with_empty_hobbies['hobbies'] = {}
        
        features = create_hobby_features(data_with_empty_hobbies['hobbies'])
        
        # Check default values for empty hobbies
        self.assertEqual(features['hobby_count'], 0)
        self.assertEqual(features['avg_intensity'], 0.0)
        self.assertEqual(features['avg_proficiency'], 0.0)
        self.assertEqual(features['avg_years'], 0.0)
        self.assertEqual(features['max_intensity'], 0.0)
        self.assertEqual(features['max_proficiency'], 0.0)
        self.assertEqual(features['max_years'], 0.0)
    
    def test_missing_academic_subjects(self):
        """Test handling of missing academic subjects"""
        incomplete_academic = {
            'mathematics': 0.85,
            'science': 0.80
            # Missing other subjects
        }
        
        features = create_academic_features(incomplete_academic)
        
        # Should handle missing subjects gracefully
        self.assertIn('mathematics', features)
        self.assertIn('science', features)
        self.assertIn('english', features)  # Should have default value
        self.assertIn('social_science', features)  # Should have default value
        self.assertIn('second_language', features)  # Should have default value


if __name__ == '__main__':
    unittest.main() 