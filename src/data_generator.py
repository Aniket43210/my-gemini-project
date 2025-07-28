import pandas as pd
import numpy as np
import random
from collections import defaultdict
from config.hobby_taxonomy import HOBBY_TAXONOMY, CAREER_HIERARCHY

class SyntheticDataGenerator:
    """Generate synthetic training data for career prediction model"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.career_templates = self._create_career_templates()
    
    def _create_career_templates(self):
        """Create templates for different career types with expected patterns"""
        templates = {
            'software_engineer': {
                'academic_preferences': {
                    'mathematics': (0.7, 0.9),
                    'science': (0.6, 0.85),
                    'english': (0.4, 0.7),
                    'social_science': (0.3, 0.6),
                    'second_language': (0.3, 0.6)
                },
                'personality_preferences': {
                    'openness': (0.6, 0.9),
                    'conscientiousness': (0.6, 0.9),
                    'extraversion': (0.3, 0.7),
                    'agreeableness': (0.4, 0.7),
                    'neuroticism': (0.2, 0.6)
                },
                'hobby_preferences': ['programming', 'robotics', 'gaming', 'research'],
                'hobby_weights': [0.9, 0.7, 0.6, 0.5]
            },
            'data_scientist': {
                'academic_preferences': {
                    'mathematics': (0.8, 0.95),
                    'science': (0.7, 0.9),
                    'english': (0.5, 0.8),
                    'social_science': (0.4, 0.7),
                    'second_language': (0.3, 0.6)
                },
                'personality_preferences': {
                    'openness': (0.7, 0.95),
                    'conscientiousness': (0.7, 0.9),
                    'extraversion': (0.3, 0.7),
                    'agreeableness': (0.4, 0.7),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['programming', 'research', 'reading', 'investing'],
                'hobby_weights': [0.8, 0.9, 0.7, 0.6]
            },
            'graphic_designer': {
                'academic_preferences': {
                    'mathematics': (0.3, 0.6),
                    'science': (0.3, 0.6),
                    'english': (0.6, 0.9),
                    'social_science': (0.5, 0.8),
                    'second_language': (0.4, 0.8)
                },
                'personality_preferences': {
                    'openness': (0.8, 0.95),
                    'conscientiousness': (0.5, 0.8),
                    'extraversion': (0.4, 0.8),
                    'agreeableness': (0.5, 0.8),
                    'neuroticism': (0.3, 0.7)
                },
                'hobby_preferences': ['photography', 'writing', 'music', 'cooking'],
                'hobby_weights': [0.9, 0.6, 0.7, 0.5]
            },
            'teacher': {
                'academic_preferences': {
                    'mathematics': (0.5, 0.8),
                    'science': (0.5, 0.8),
                    'english': (0.7, 0.9),
                    'social_science': (0.6, 0.9),
                    'second_language': (0.5, 0.8)
                },
                'personality_preferences': {
                    'openness': (0.6, 0.9),
                    'conscientiousness': (0.7, 0.9),
                    'extraversion': (0.6, 0.9),
                    'agreeableness': (0.7, 0.95),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['volunteering', 'reading', 'public_speaking', 'writing'],
                'hobby_weights': [0.8, 0.9, 0.8, 0.7]
            },
            'financial_analyst': {
                'academic_preferences': {
                    'mathematics': (0.8, 0.95),
                    'science': (0.5, 0.8),
                    'english': (0.6, 0.8),
                    'social_science': (0.7, 0.9),
                    'second_language': (0.4, 0.7)
                },
                'personality_preferences': {
                    'openness': (0.5, 0.8),
                    'conscientiousness': (0.8, 0.95),
                    'extraversion': (0.4, 0.8),
                    'agreeableness': (0.4, 0.7),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['investing', 'reading', 'research', 'entrepreneurship'],
                'hobby_weights': [0.9, 0.8, 0.7, 0.6]
            },
            'marketing_manager': {
                'academic_preferences': {
                    'mathematics': (0.5, 0.7),
                    'science': (0.3, 0.6),
                    'english': (0.7, 0.9),
                    'social_science': (0.7, 0.9),
                    'second_language': (0.6, 0.9)
                },
                'personality_preferences': {
                    'openness': (0.7, 0.9),
                    'conscientiousness': (0.6, 0.8),
                    'extraversion': (0.7, 0.95),
                    'agreeableness': (0.6, 0.8),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['public_speaking', 'writing', 'photography', 'entrepreneurship'],
                'hobby_weights': [0.8, 0.7, 0.6, 0.7]
            },
            'mechanical_engineer': {
                'academic_preferences': {
                    'mathematics': (0.8, 0.95),
                    'science': (0.8, 0.95),
                    'english': (0.4, 0.7),
                    'social_science': (0.3, 0.6),
                    'second_language': (0.3, 0.6)
                },
                'personality_preferences': {
                    'openness': (0.6, 0.8),
                    'conscientiousness': (0.7, 0.9),
                    'extraversion': (0.3, 0.7),
                    'agreeableness': (0.4, 0.7),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['robotics', 'team_sports', 'individual_sports', 'cooking'],
                'hobby_weights': [0.8, 0.6, 0.5, 0.4]
            },
            'doctor': {
                'academic_preferences': {
                    'mathematics': (0.7, 0.9),
                    'science': (0.9, 0.98),
                    'english': (0.6, 0.8),
                    'social_science': (0.5, 0.8),
                    'second_language': (0.4, 0.7)
                },
                'personality_preferences': {
                    'openness': (0.6, 0.8),
                    'conscientiousness': (0.8, 0.95),
                    'extraversion': (0.5, 0.8),
                    'agreeableness': (0.7, 0.9),
                    'neuroticism': (0.2, 0.4)
                },
                'hobby_preferences': ['volunteering', 'research', 'reading', 'individual_sports'],
                'hobby_weights': [0.8, 0.7, 0.8, 0.6]
            },
            'lawyer': {
                'academic_preferences': {
                    'mathematics': (0.5, 0.7),
                    'science': (0.4, 0.6),
                    'english': (0.8, 0.95),
                    'social_science': (0.8, 0.95),
                    'second_language': (0.5, 0.8)
                },
                'personality_preferences': {
                    'openness': (0.6, 0.8),
                    'conscientiousness': (0.7, 0.9),
                    'extraversion': (0.6, 0.9),
                    'agreeableness': (0.4, 0.7),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['public_speaking', 'reading', 'writing', 'volunteering'],
                'hobby_weights': [0.9, 0.9, 0.8, 0.6]
            },
            'chef': {
                'academic_preferences': {
                    'mathematics': (0.4, 0.6),
                    'science': (0.5, 0.7),
                    'english': (0.5, 0.7),
                    'social_science': (0.4, 0.6),
                    'second_language': (0.5, 0.8)
                },
                'personality_preferences': {
                    'openness': (0.7, 0.9),
                    'conscientiousness': (0.6, 0.8),
                    'extraversion': (0.5, 0.8),
                    'agreeableness': (0.5, 0.8),
                    'neuroticism': (0.3, 0.6)
                },
                'hobby_preferences': ['cooking', 'entrepreneurship', 'team_sports', 'music'],
                'hobby_weights': [0.95, 0.6, 0.5, 0.6]
            },
            'web_developer': {
                'academic_preferences': {
                    'mathematics': (0.7, 0.9),
                    'science': (0.6, 0.8),
                    'english': (0.5, 0.8),
                    'social_science': (0.3, 0.6),
                    'second_language': (0.4, 0.7)
                },
                'personality_preferences': {
                    'openness': (0.7, 0.9),
                    'conscientiousness': (0.6, 0.9),
                    'extraversion': (0.3, 0.7),
                    'agreeableness': (0.4, 0.7),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['programming', 'photography', 'gaming', 'music'],
                'hobby_weights': [0.95, 0.6, 0.7, 0.5]
            },
            'ux_designer': {
                'academic_preferences': {
                    'mathematics': (0.5, 0.7),
                    'science': (0.4, 0.6),
                    'english': (0.7, 0.9),
                    'social_science': (0.6, 0.8),
                    'second_language': (0.5, 0.8)
                },
                'personality_preferences': {
                    'openness': (0.8, 0.95),
                    'conscientiousness': (0.6, 0.8),
                    'extraversion': (0.5, 0.8),
                    'agreeableness': (0.6, 0.9),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['photography', 'programming', 'research', 'writing'],
                'hobby_weights': [0.8, 0.7, 0.6, 0.5]
            },
            'nurse': {
                'academic_preferences': {
                    'mathematics': (0.6, 0.8),
                    'science': (0.8, 0.95),
                    'english': (0.6, 0.8),
                    'social_science': (0.5, 0.8),
                    'second_language': (0.4, 0.7)
                },
                'personality_preferences': {
                    'openness': (0.5, 0.8),
                    'conscientiousness': (0.7, 0.9),
                    'extraversion': (0.5, 0.8),
                    'agreeableness': (0.8, 0.95),
                    'neuroticism': (0.2, 0.4)
                },
                'hobby_preferences': ['volunteering', 'individual_sports', 'reading', 'cooking'],
                'hobby_weights': [0.9, 0.6, 0.7, 0.5]
            },
            'accountant': {
                'academic_preferences': {
                    'mathematics': (0.8, 0.95),
                    'science': (0.4, 0.6),
                    'english': (0.6, 0.8),
                    'social_science': (0.5, 0.7),
                    'second_language': (0.3, 0.6)
                },
                'personality_preferences': {
                    'openness': (0.4, 0.7),
                    'conscientiousness': (0.8, 0.95),
                    'extraversion': (0.3, 0.6),
                    'agreeableness': (0.5, 0.7),
                    'neuroticism': (0.2, 0.5)
                },
                'hobby_preferences': ['reading', 'individual_sports', 'investing', 'gaming'],
                'hobby_weights': [0.7, 0.6, 0.8, 0.4]
            },
            'psychologist': {
                'academic_preferences': {
                    'mathematics': (0.5, 0.7),
                    'science': (0.6, 0.8),
                    'english': (0.8, 0.95),
                    'social_science': (0.8, 0.95),
                    'second_language': (0.5, 0.8)
                },
                'personality_preferences': {
                    'openness': (0.8, 0.95),
                    'conscientiousness': (0.7, 0.9),
                    'extraversion': (0.6, 0.9),
                    'agreeableness': (0.8, 0.95),
                    'neuroticism': (0.2, 0.4)
                },
                'hobby_preferences': ['research', 'reading', 'volunteering', 'writing'],
                'hobby_weights': [0.8, 0.9, 0.8, 0.6]
            }
        }
        return templates
    
    def generate_user_profile(self, career):
        """Generate a single user profile for a specific career"""
        if career not in self.career_templates:
            raise ValueError(f"Career '{career}' not found in templates")
        
        template = self.career_templates[career]
        
        # Generate academic grades with some noise
        academic_grades = {}
        for subject, (min_val, max_val) in template['academic_preferences'].items():
            # Add random noise but keep within bounds
            base_score = np.random.uniform(min_val, max_val)
            noise = np.random.normal(0, 0.05)  # Small noise
            final_score = np.clip(base_score + noise, 0.1, 1.0)
            academic_grades[subject] = round(final_score, 2)
        
        # Generate personality scores
        personality_scores = {}
        for trait, (min_val, max_val) in template['personality_preferences'].items():
            base_score = np.random.uniform(min_val, max_val)
            noise = np.random.normal(0, 0.05)
            final_score = np.clip(base_score + noise, 0.1, 1.0)
            personality_scores[trait] = round(final_score, 2)
        
        # Generate hobbies with detailed attributes
        user_hobbies = {}
        hobby_prefs = template['hobby_preferences']
        hobby_weights = template['hobby_weights']
        
        # Select 2-4 hobbies for this user
        num_hobbies = np.random.randint(2, 5)
        selected_hobbies = np.random.choice(
            hobby_prefs, 
            size=min(num_hobbies, len(hobby_prefs)), 
            replace=False,
            p=np.array(hobby_weights) / np.sum(hobby_weights)
        )
        
        for hobby in selected_hobbies:
            base_intensity = hobby_weights[hobby_prefs.index(hobby)]
            intensity_noise = np.random.normal(0, 0.1)
            intensity = np.clip(base_intensity + intensity_noise, 0.2, 1.0)
            
            # Generate proficiency and years with some correlation to intensity
            proficiency = np.clip(intensity + np.random.normal(0, 0.15), 0.2, 1.0)
            years = max(1, int(np.random.exponential(3) + 1))
            
            user_hobbies[hobby] = {
                'intensity': round(intensity, 2),
                'proficiency': round(proficiency, 2),
                'years': years
            }
        
        # Sometimes add a random hobby for diversity
        if np.random.random() < 0.3:
            all_hobbies = list(HOBBY_TAXONOMY.keys())
            random_hobby = np.random.choice([h for h in all_hobbies if h not in user_hobbies])
            user_hobbies[random_hobby] = {
                'intensity': round(np.random.uniform(0.2, 0.6), 2),
                'proficiency': round(np.random.uniform(0.2, 0.6), 2),
                'years': np.random.randint(1, 4)
            }
        
        return {
            'academic_grades': academic_grades,
            'hobbies': user_hobbies,
            'personality': personality_scores,
            'career': career
        }
    
    def generate_dataset(self, samples_per_career=100):
        """Generate a complete synthetic dataset"""
        dataset = []
        careers = list(self.career_templates.keys())
        
        print(f"Generating {samples_per_career} samples for each of {len(careers)} careers...")
        
        for career in careers:
            print(f"Generating data for {career}...")
            for _ in range(samples_per_career):
                try:
                    user_profile = self.generate_user_profile(career)
                    dataset.append(user_profile)
                except Exception as e:
                    print(f"Error generating profile for {career}: {e}")
                    continue
        
        print(f"Generated {len(dataset)} total samples")
        return dataset
    
    def save_dataset(self, dataset, filepath):
        """Save dataset to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath):
        """Load dataset from JSON file"""
        import json
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        print(f"Dataset loaded from {filepath}")
        return dataset

# Usage example
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
    # Generate a small dataset for testing
    dataset = generator.generate_dataset(samples_per_career=50)
    
    # Save dataset
    generator.save_dataset(dataset, "data/synthetic_career_data.json")
    
    # Show sample
    print("\nSample user profile:")
    sample = dataset[0]
    print(f"Career: {sample['career']}")
    print(f"Academic grades: {sample['academic_grades']}")
    print(f"Personality: {sample['personality']}")
    print(f"Hobbies: {sample['hobbies']}")
