"""
Unit tests for config/hobby_taxonomy.py
Tests the configuration and taxonomy functionality
"""

import unittest
import sys
import os

# Add config to path
sys.path.append('config')

from config.hobby_taxonomy import (
    CAREER_HIERARCHY, HOBBY_TAXONOMY,
    get_career_field, get_career_broad_category,
    get_hobby_relevance, get_all_careers, get_all_hobbies
)


class TestHobbyTaxonomy(unittest.TestCase):
    """Unit tests for hobby taxonomy configuration"""
    
    def test_career_hierarchy_structure(self):
        """Test career hierarchy structure"""
        self.assertIsInstance(CAREER_HIERARCHY, dict)
        self.assertGreater(len(CAREER_HIERARCHY), 0)
        
        # Check each career has required fields
        for career, info in CAREER_HIERARCHY.items():
            self.assertIsInstance(career, str)
            self.assertIsInstance(info, dict)
            
            # Check required fields
            self.assertIn('field', info)
            self.assertIn('broad_category', info)
            self.assertIn('skills', info)
            
            # Check data types
            self.assertIsInstance(info['field'], str)
            self.assertIsInstance(info['broad_category'], str)
            self.assertIsInstance(info['skills'], list)
            
            # Check skills list
            self.assertGreater(len(info['skills']), 0)
            for skill in info['skills']:
                self.assertIsInstance(skill, str)
    
    def test_hobby_taxonomy_structure(self):
        """Test hobby taxonomy structure"""
        self.assertIsInstance(HOBBY_TAXONOMY, dict)
        self.assertGreater(len(HOBBY_TAXONOMY), 0)
        
        # Check each hobby has required fields
        for hobby, info in HOBBY_TAXONOMY.items():
            self.assertIsInstance(hobby, str)
            self.assertIsInstance(info, dict)
            
            # Check required fields
            self.assertIn('category', info)
            self.assertIn('skills', info)
            self.assertIn('career_relevance', info)
            
            # Check data types
            self.assertIsInstance(info['category'], str)
            self.assertIsInstance(info['skills'], list)
            self.assertIsInstance(info['career_relevance'], dict)
            
            # Check skills list
            self.assertGreater(len(info['skills']), 0)
            for skill in info['skills']:
                self.assertIsInstance(skill, str)
            
            # Check career relevance
            for career, relevance in info['career_relevance'].items():
                self.assertIsInstance(career, str)
                self.assertIsInstance(relevance, float)
                self.assertGreaterEqual(relevance, 0.0)
                self.assertLessEqual(relevance, 1.0)
    
    def test_get_career_field(self):
        """Test get_career_field function"""
        # Test valid careers
        self.assertEqual(get_career_field('software_engineer'), 'engineering')
        self.assertEqual(get_career_field('data_scientist'), 'data_science')
        self.assertEqual(get_career_field('teacher'), 'education')
        
        # Test invalid career
        with self.assertRaises(KeyError):
            get_career_field('invalid_career')
    
    def test_get_career_broad_category(self):
        """Test get_career_broad_category function"""
        # Test valid careers
        self.assertEqual(get_career_broad_category('software_engineer'), 'STEM')
        self.assertEqual(get_career_broad_category('financial_analyst'), 'Business')
        self.assertEqual(get_career_broad_category('teacher'), 'Social')
        
        # Test invalid career
        with self.assertRaises(KeyError):
            get_career_broad_category('invalid_career')
    
    def test_get_hobby_relevance(self):
        """Test get_hobby_relevance function"""
        # Test valid hobby-career combinations
        relevance = get_hobby_relevance('programming', 'software_engineer')
        self.assertIsInstance(relevance, float)
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)
        
        # Test hobby not in taxonomy
        with self.assertRaises(KeyError):
            get_hobby_relevance('invalid_hobby', 'software_engineer')
        
        # Test career not in hobby relevance
        with self.assertRaises(KeyError):
            get_hobby_relevance('programming', 'invalid_career')
    
    def test_get_all_careers(self):
        """Test get_all_careers function"""
        careers = get_all_careers()
        
        # Check structure
        self.assertIsInstance(careers, list)
        self.assertGreater(len(careers), 0)
        
        # Check all careers are strings
        for career in careers:
            self.assertIsInstance(career, str)
        
        # Check all careers are in hierarchy
        for career in careers:
            self.assertIn(career, CAREER_HIERARCHY)
    
    def test_get_all_hobbies(self):
        """Test get_all_hobbies function"""
        hobbies = get_all_hobbies()
        
        # Check structure
        self.assertIsInstance(hobbies, list)
        self.assertGreater(len(hobbies), 0)
        
        # Check all hobbies are strings
        for hobby in hobbies:
            self.assertIsInstance(hobby, str)
        
        # Check all hobbies are in taxonomy
        for hobby in hobbies:
            self.assertIn(hobby, HOBBY_TAXONOMY)
    
    def test_career_consistency(self):
        """Test consistency between career hierarchy and hobby taxonomy"""
        careers = get_all_careers()
        hobbies = get_all_hobbies()
        
        # Check that all careers mentioned in hobby relevance exist in hierarchy
        for hobby, info in HOBBY_TAXONOMY.items():
            for career in info['career_relevance'].keys():
                self.assertIn(career, careers, f"Career {career} in hobby {hobby} not found in hierarchy")
    
    def test_broad_categories_consistency(self):
        """Test that broad categories are consistent"""
        broad_categories = set()
        
        for career, info in CAREER_HIERARCHY.items():
            broad_categories.add(info['broad_category'])
        
        # Check that we have a reasonable number of broad categories
        self.assertGreater(len(broad_categories), 3)
        self.assertLess(len(broad_categories), 10)
        
        # Check that all broad categories are valid
        valid_categories = {
            'STEM', 'Business', 'Creative', 'Social', 
            'Healthcare', 'Law_Government'
        }
        
        for category in broad_categories:
            self.assertIn(category, valid_categories)
    
    def test_field_consistency(self):
        """Test that fields are consistent"""
        fields = set()
        
        for career, info in CAREER_HIERARCHY.items():
            fields.add(info['field'])
        
        # Check that we have a reasonable number of fields
        self.assertGreater(len(fields), 5)
        self.assertLess(len(fields), 20)
        
        # Check that all fields are valid
        valid_fields = {
            'engineering', 'data_science', 'design', 'education', 
            'finance', 'marketing', 'psychology', 'medicine', 
            'healthcare', 'legal', 'culinary'
        }
        
        for field in fields:
            self.assertIn(field, valid_fields)
    
    def test_hobby_categories_consistency(self):
        """Test that hobby categories are consistent"""
        categories = set()
        
        for hobby, info in HOBBY_TAXONOMY.items():
            categories.add(info['category'])
        
        # Check that we have a reasonable number of categories
        self.assertGreater(len(categories), 1)
        self.assertLess(len(categories), 10)
        
        # Check that all categories are valid
        valid_categories = {
            'technical', 'analytical', 'creative', 'physical',
            'intellectual', 'social'
        }
        
        for category in categories:
            self.assertIn(category.lower(), {c.lower() for c in valid_categories})
    
    def test_skills_consistency(self):
        """Test that skills are consistent across careers and hobbies"""
        career_skills = set()
        hobby_skills = set()
        
        # Collect all skills from careers
        for career, info in CAREER_HIERARCHY.items():
            career_skills.update(info['skills'])
        
        # Collect all skills from hobbies
        for hobby, info in HOBBY_TAXONOMY.items():
            hobby_skills.update(info['skills'])
        
        # Check that skills are not empty
        self.assertGreater(len(career_skills), 0)
        self.assertGreater(len(hobby_skills), 0)
        
        # Check that all skills are strings
        for skill in career_skills:
            self.assertIsInstance(skill, str)
        
        for skill in hobby_skills:
            self.assertIsInstance(skill, str)
    
    def test_relevance_scores_consistency(self):
        """Test that relevance scores are consistent"""
        for hobby, info in HOBBY_TAXONOMY.items():
            for career, relevance in info['career_relevance'].items():
                # Check relevance is a valid float
                self.assertIsInstance(relevance, float)
                self.assertGreaterEqual(relevance, 0.0)
                self.assertLessEqual(relevance, 1.0)
                
                # Check that relevance is not exactly 0 for all careers
                # (at least one career should have some relevance)
                if all(rel == 0.0 for rel in info['career_relevance'].values()):
                    self.fail(f"Hobby {hobby} has zero relevance for all careers")


if __name__ == '__main__':
    unittest.main() 