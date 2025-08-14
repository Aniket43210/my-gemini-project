"""
Unit tests for data_generator.py
Tests the SyntheticDataGenerator class and its methods
"""

import unittest
import sys
import os
import json
import tempfile
import shutil

# Add src and config to path
sys.path.append('src')
sys.path.append('config')

from src.data_generator import SyntheticDataGenerator
from config.hobby_taxonomy import CAREER_HIERARCHY, HOBBY_TAXONOMY


class TestSyntheticDataGenerator(unittest.TestCase):
    """Unit tests for SyntheticDataGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = SyntheticDataGenerator(seed=42)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test generator initialization"""
        self.assertIsNotNone(self.generator)
        self.assertIsNotNone(self.generator.career_templates)
        self.assertGreater(len(self.generator.career_templates), 0)
    
    def test_career_templates_structure(self):
        """Test career templates have correct structure"""
        for career, template in self.generator.career_templates.items():
            # Check required keys exist
            self.assertIn('academic_preferences', template)
            self.assertIn('personality_preferences', template)
            self.assertIn('hobby_preferences', template)
            self.assertIn('hobby_weights', template)
            
            # Check academic preferences structure
            academic = template['academic_preferences']
            required_subjects = ['mathematics', 'science', 'english', 'social_science', 'second_language']
            for subject in required_subjects:
                self.assertIn(subject, academic)
                self.assertIsInstance(academic[subject], tuple)
                self.assertEqual(len(academic[subject]), 2)
                self.assertLess(academic[subject][0], academic[subject][1])
                self.assertGreaterEqual(academic[subject][0], 0.0)
                self.assertLessEqual(academic[subject][1], 1.0)
            
            # Check personality preferences structure
            personality = template['personality_preferences']
            required_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            for trait in required_traits:
                self.assertIn(trait, personality)
                self.assertIsInstance(personality[trait], tuple)
                self.assertEqual(len(personality[trait]), 2)
                self.assertLess(personality[trait][0], personality[trait][1])
                self.assertGreaterEqual(personality[trait][0], 0.0)
                self.assertLessEqual(personality[trait][1], 1.0)
            
            # Check hobby preferences
            self.assertIsInstance(template['hobby_preferences'], list)
            self.assertIsInstance(template['hobby_weights'], list)
            self.assertEqual(len(template['hobby_preferences']), len(template['hobby_weights']))
    
    def test_generate_user_profile_structure(self):
        """Test user profile generation structure"""
        test_career = 'software_engineer'
        profile = self.generator.generate_user_profile(test_career)
        
        # Check profile structure
        self.assertIn('academic_grades', profile)
        self.assertIn('personality', profile)
        self.assertIn('hobbies', profile)
        self.assertIn('career', profile)
        
        # Check career assignment
        self.assertEqual(profile['career'], test_career)
        
        # Check academic grades
        academic = profile['academic_grades']
        required_subjects = ['mathematics', 'science', 'english', 'social_science', 'second_language']
        for subject in required_subjects:
            self.assertIn(subject, academic)
            self.assertIsInstance(academic[subject], float)
            self.assertGreaterEqual(academic[subject], 0.0)
            self.assertLessEqual(academic[subject], 1.0)
        
        # Check personality traits
        personality = profile['personality']
        required_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in required_traits:
            self.assertIn(trait, personality)
            self.assertIsInstance(personality[trait], float)
            self.assertGreaterEqual(personality[trait], 0.0)
            self.assertLessEqual(personality[trait], 1.0)
        
        # Check hobbies
        hobbies = profile['hobbies']
        self.assertIsInstance(hobbies, dict)
        for hobby_name, hobby_data in hobbies.items():
            self.assertIn('intensity', hobby_data)
            self.assertIn('proficiency', hobby_data)
            self.assertIn('years', hobby_data)
            self.assertIsInstance(hobby_data['intensity'], float)
            self.assertIsInstance(hobby_data['proficiency'], float)
            self.assertIsInstance(hobby_data['years'], int)
            self.assertGreaterEqual(hobby_data['intensity'], 0.0)
            self.assertLessEqual(hobby_data['intensity'], 1.0)
            self.assertGreaterEqual(hobby_data['proficiency'], 0.0)
            self.assertLessEqual(hobby_data['proficiency'], 1.0)
            self.assertGreaterEqual(hobby_data['years'], 0)
    
    def test_generate_small_dataset(self):
        """Test dataset generation with minimal samples"""
        samples_per_career = 2
        dataset = self.generator.generate_dataset(samples_per_career=samples_per_career)
        
        # Check dataset structure
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        # Check each sample structure
        for sample in dataset:
            self.assertIn('academic_grades', sample)
            self.assertIn('personality', sample)
            self.assertIn('hobbies', sample)
            self.assertIn('career', sample)
        
        # Check career distribution
        career_counts = {}
        for sample in dataset:
            career = sample['career']
            career_counts[career] = career_counts.get(career, 0) + 1
        
        # Should have samples for each career
        expected_careers = set(self.generator.career_templates.keys())
        actual_careers = set(career_counts.keys())
        self.assertEqual(expected_careers, actual_careers)
    
    def test_save_and_load_dataset(self):
        """Test dataset saving and loading"""
        # Generate minimal test dataset
        dataset = self.generator.generate_dataset(samples_per_career=1)
        
        # Save dataset
        filepath = os.path.join(self.temp_dir, 'test_dataset.json')
        self.generator.save_dataset(dataset, filepath)
        
        # Check file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Load dataset
        loaded_dataset = self.generator.load_dataset(filepath)
        
        # Check loaded dataset matches original
        self.assertEqual(len(dataset), len(loaded_dataset))
        
        # Check structure of loaded data
        for i, (original, loaded) in enumerate(zip(dataset, loaded_dataset)):
            self.assertEqual(original['career'], loaded['career'])
            self.assertEqual(original['academic_grades'], loaded['academic_grades'])
            self.assertEqual(original['personality'], loaded['personality'])
            self.assertEqual(original['hobbies'], loaded['hobbies'])
    
    def test_invalid_career_handling(self):
        """Test handling of invalid career names"""
        with self.assertRaises(KeyError):
            self.generator.generate_user_profile('invalid_career')
    
    def test_reproducibility(self):
        """Test that same seed produces similar results within expected ranges"""
        generator1 = SyntheticDataGenerator(seed=42)
        generator2 = SyntheticDataGenerator(seed=42)
        
        profile1 = generator1.generate_user_profile('software_engineer')
        profile2 = generator2.generate_user_profile('software_engineer')
        
        # Check that values are within template ranges
        template = generator1.career_templates['software_engineer']
        
        for subject, (min_val, max_val) in template['academic_preferences'].items():
            self.assertGreaterEqual(profile1['academic_grades'][subject], min_val)
            self.assertLessEqual(profile1['academic_grades'][subject], max_val)
            self.assertGreaterEqual(profile2['academic_grades'][subject], min_val)
            self.assertLessEqual(profile2['academic_grades'][subject], max_val)
        
        for trait, (min_val, max_val) in template['personality_preferences'].items():
            self.assertGreaterEqual(profile1['personality'][trait], min_val)
            self.assertLessEqual(profile1['personality'][trait], max_val)
            self.assertGreaterEqual(profile2['personality'][trait], min_val)
            self.assertLessEqual(profile2['personality'][trait], max_val)
        
        # Check that hobbies are from the template's preferences
        hobby_prefs = set(template['hobby_preferences'])
        self.assertTrue(all(hobby in hobby_prefs for hobby in profile1['hobbies']))
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        generator1 = SyntheticDataGenerator(seed=42)
        generator2 = SyntheticDataGenerator(seed=123)
        
        profile1 = generator1.generate_user_profile('software_engineer')
        profile2 = generator2.generate_user_profile('software_engineer')
        
        # At least some values should be different
        self.assertNotEqual(profile1['academic_grades'], profile2['academic_grades'])


if __name__ == '__main__':
    unittest.main() 