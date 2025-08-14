"""
Integration tests for main.py
Tests the main application functionality without large data generation
"""

import unittest
import sys
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

# Add src and config to path
sys.path.append('src')
sys.path.append('config')

# Import main functions
import main


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main application"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Create minimal test data
        self.test_data = [
            {
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
                    'reading': {'intensity': 0.7, 'proficiency': 0.6, 'years': 2}
                },
                'career': 'software_engineer'
            },
            {
                'academic_grades': {
                    'mathematics': 0.90,
                    'science': 0.85,
                    'english': 0.70,
                    'social_science': 0.55,
                    'second_language': 0.60
                },
                'personality': {
                    'openness': 0.90,
                    'conscientiousness': 0.80,
                    'extraversion': 0.40,
                    'agreeableness': 0.60,
                    'neuroticism': 0.30
                },
                'hobbies': {
                    'programming': {'intensity': 0.95, 'proficiency': 0.85, 'years': 5},
                    'research': {'intensity': 0.8, 'proficiency': 0.7, 'years': 3}
                },
                'career': 'data_scientist'
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
        os.chdir(self.original_cwd)
    
    def test_safe_float_function(self):
        """Test safe_float utility function"""
        # Test valid float
        self.assertEqual(main.safe_float(0.5), 0.5)
        self.assertEqual(main.safe_float("0.7"), 0.7)
        
        # Test None and invalid values
        self.assertEqual(main.safe_float(None), 0.5)  # default
        self.assertEqual(main.safe_float("invalid"), 0.5)  # default
        self.assertEqual(main.safe_float("invalid", 0.3), 0.3)  # custom default
    
    def test_create_directories(self):
        """Test directory creation"""
        test_dirs = ['test_data', 'test_models', 'test_results', 'test_logs']
        
        # Create directories in temp location
        os.chdir(self.temp_dir)
        main.create_directories()
        
        # Check directories were created
        for directory in test_dirs:
            self.assertTrue(os.path.exists(directory))
    
    def test_clean_and_validate_data(self):
        """Test data cleaning and validation"""
        cleaned_data = main.clean_and_validate_data(self.test_data)
        
        # Check data structure
        self.assertIsInstance(cleaned_data, list)
        self.assertEqual(len(cleaned_data), len(self.test_data))
        
        # Check each sample has required fields
        for sample in cleaned_data:
            self.assertIn('academic_grades', sample)
            self.assertIn('personality', sample)
            self.assertIn('hobbies', sample)
            self.assertIn('career', sample)
            
            # Check academic grades
            academic = sample['academic_grades']
            required_subjects = ['mathematics', 'science', 'english', 'social_science', 'second_language']
            for subject in required_subjects:
                self.assertIn(subject, academic)
                self.assertIsInstance(academic[subject], float)
                self.assertGreaterEqual(academic[subject], 0.0)
                self.assertLessEqual(academic[subject], 1.0)
            
            # Check personality traits
            personality = sample['personality']
            required_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            for trait in required_traits:
                self.assertIn(trait, personality)
                self.assertIsInstance(personality[trait], float)
                self.assertGreaterEqual(personality[trait], 0.0)
                self.assertLessEqual(personality[trait], 1.0)
    
    def test_create_ultimate_features(self):
        """Test feature creation from raw data"""
        features_list = main.create_ultimate_features(self.test_data)
        
        # Check features structure
        self.assertIsInstance(features_list, list)
        self.assertEqual(len(features_list), len(self.test_data))
        
        # Check each feature set
        for features in features_list:
            self.assertIsInstance(features, dict)
            
            # Check that we have a reasonable number of features
            self.assertGreater(len(features), 40)
            self.assertLess(len(features), 60)
            
            # Check some key features exist
            self.assertIn('mathematics', features)
            self.assertIn('openness', features)
            self.assertIn('hobby_count', features)
            self.assertIn('academic_peak', features)
            
            # Check all features are numeric
            for feature_name, feature_value in features.items():
                self.assertIsInstance(feature_value, (int, float, bool))
                if isinstance(feature_value, (int, float)) and not isinstance(feature_value, bool):
                    self.assertGreaterEqual(feature_value, 0.0)
                    self.assertLessEqual(feature_value, 1.0)
    
    def test_create_hierarchical_labels(self):
        """Test hierarchical label creation"""
        career_labels = [sample['career'] for sample in self.test_data]
        
        broad_labels, field_labels = main.create_hierarchical_labels(career_labels)
        
        # Check label structure
        self.assertIsInstance(broad_labels, list)
        self.assertIsInstance(field_labels, list)
        self.assertEqual(len(broad_labels), len(career_labels))
        self.assertEqual(len(field_labels), len(career_labels))
        
        # Check label values
        for broad_label in broad_labels:
            self.assertIsInstance(broad_label, str)
            self.assertIn(broad_label, ['STEM', 'Business', 'Creative', 'Social', 'Healthcare', 'Law_Government'])
        
        for field_label in field_labels:
            self.assertIsInstance(field_label, str)
            self.assertIn(field_label, ['engineering', 'data_science', 'design', 'education', 'finance', 'marketing', 'psychology', 'medicine', 'healthcare', 'legal', 'culinary'])
    
    @patch('main.train_hierarchical_models')
    def test_main_training_flow(self, mock_train):
        """Test main training flow without actual model training"""
        # Mock the training function to return test results
        mock_results = {
            'broad_model': MagicMock(),
            'field_model': MagicMock(),
            'career_model': MagicMock(),
            'broad_encoder': MagicMock(),
            'field_encoder': MagicMock(),
            'career_encoder': MagicMock()
        }
        mock_train.return_value = mock_results
        
        # Test the main function with minimal data
        with patch('main.load_and_prepare_data', return_value=self.test_data):
            with patch('main.create_ultimate_features', return_value=[[0.5]*50, [0.6]*50]):
                with patch('main.create_hierarchical_labels', return_value=(['STEM', 'STEM'], ['engineering', 'data_science'])):
                    with patch('main.save_ultimate_results'):
                        # This should run without errors
                        try:
                            main.main()
                        except Exception as e:
                            # If there are any issues, they should be caught and handled gracefully
                            self.fail(f"Main function failed with error: {e}")
    
    def test_create_ultimate_predictor(self):
        """Test predictor creation"""
        # Create mock results
        mock_results = {
            'broad_model': MagicMock(),
            'field_model': MagicMock(),
            'career_model': MagicMock(),
            'broad_encoder': MagicMock(),
            'field_encoder': MagicMock(),
            'career_encoder': MagicMock()
        }
        
        # Mock the encoders
        mock_results['broad_encoder'].transform.return_value = [0]
        mock_results['field_encoder'].transform.return_value = [0]
        mock_results['career_encoder'].inverse_transform.return_value = ['software_engineer']
        
        # Mock the models
        mock_results['broad_model'].predict_proba.return_value = [[0.1, 0.9]]
        mock_results['field_model'].predict_proba.return_value = [[0.2, 0.8]]
        mock_results['career_model'].predict_proba.return_value = [[0.3, 0.7]]
        
        # Create predictor
        predictor = main.create_ultimate_predictor(mock_results)
        
        # Test prediction
        test_input = {
            'academic_grades': {
                'mathematics': 0.85,
                'science': 0.80,
                'english': 0.75,
                'social_science': 0.60,
                'second_language': 0.65
            },
            'hobbies': {
                'programming': {'intensity': 0.9, 'proficiency': 0.8, 'years': 4}
            },
            'personality': {
                'openness': 0.85,
                'conscientiousness': 0.75,
                'extraversion': 0.45,
                'agreeableness': 0.65,
                'neuroticism': 0.35
            }
        }
        
        prediction = predictor.predict_user_career(**test_input)
        
        # Check prediction structure
        self.assertIsInstance(prediction, dict)
        self.assertIn('primary_recommendation', prediction)
        self.assertIn('alternative_recommendations', prediction)
        self.assertIn('confidence_scores', prediction)
        self.assertIn('feature_importance', prediction)
        
        # Check primary recommendation
        primary = prediction['primary_recommendation']
        self.assertIn('career', primary)
        self.assertIn('confidence', primary)
        self.assertIsInstance(primary['career'], str)
        self.assertIsInstance(primary['confidence'], float)
        self.assertGreaterEqual(primary['confidence'], 0.0)
        self.assertLessEqual(primary['confidence'], 1.0)
    
    def test_data_validation_edge_cases(self):
        """Test data validation with edge cases"""
        # Test with missing fields
        incomplete_data = [
            {
                'academic_grades': {'mathematics': 0.8},  # Missing other subjects
                'personality': {'openness': 0.7},  # Missing other traits
                'hobbies': {},  # Empty hobbies
                'career': 'software_engineer'
            }
        ]
        
        cleaned_data = main.clean_and_validate_data(incomplete_data)
        
        # Should handle missing data gracefully
        self.assertEqual(len(cleaned_data), 1)
        sample = cleaned_data[0]
        
        # Check that missing fields are filled with defaults
        academic = sample['academic_grades']
        required_subjects = ['mathematics', 'science', 'english', 'social_science', 'second_language']
        for subject in required_subjects:
            self.assertIn(subject, academic)
            self.assertIsInstance(academic[subject], float)
        
        personality = sample['personality']
        required_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in required_traits:
            self.assertIn(trait, personality)
            self.assertIsInstance(personality[trait], float)
    
    def test_feature_engineering_consistency(self):
        """Test that feature engineering produces consistent results"""
        features1 = main.create_ultimate_features(self.test_data)
        features2 = main.create_ultimate_features(self.test_data)
        
        # Should produce identical results for same input
        self.assertEqual(len(features1), len(features2))
        
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            self.assertEqual(f1.keys(), f2.keys())
            for key in f1.keys():
                self.assertEqual(f1[key], f2[key], f"Feature {key} differs for sample {i}")


if __name__ == '__main__':
    unittest.main() 