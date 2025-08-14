import random
import numpy as np
import pandas as pd
from config.hobby_taxonomy import HOBBY_TAXONOMY
from collections import defaultdict

class DataAugmentationEnhancer:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
    def add_noise_to_grades(self, grades, noise_level=0.05):
        """Add Gaussian noise to academic grades"""
        noisy_grades = {}
        for subject, grade in grades.items():
            noise = np.random.normal(0, noise_level)
            noisy_grade = np.clip(grade + noise, 0, 1)
            noisy_grades[subject] = round(noisy_grade, 2)
        return noisy_grades
    
    def add_variation_to_hobbies(self, hobbies):
        """Randomly adjust hobby intensities and experience"""
        varied_hobbies = {}  
        for hobby, details in hobbies.items():
            # Add some random variation
            intensity_noise = np.random.normal(0, 0.1)
            intensity = np.clip(details['intensity'] + intensity_noise, 0.2, 1.0)
            
            # Modify proficiency in correlation to intensity change
            proficiency = np.clip(intensity + np.random.normal(0, 0.15), 0.2, 1.0)
            
            # Alter years with randomness: assume +1/-1 year adjustment
            years = max(1, details['years'] + np.random.choice([-1, 0, 1]))
            
            varied_hobbies[hobby] = {
                'intensity': round(intensity, 2),
                'proficiency': round(proficiency, 2),
                'years': years
            }
        return varied_hobbies
    
    def augment_personality_scores(self, personality, noise_factor=0.05):
        """Introduce noise into personality scores within limits"""
        varied_personality = {}
        for trait, score in personality.items():
            noise = np.random.normal(0, noise_factor)
            varied_score = np.clip(score + noise, 0, 1)
            varied_personality[trait] = round(varied_score, 2)
        return varied_personality

    def generate_augmented_dataset(self, original_dataset, augmentation_factor=1):
        """Generate augmented dataset using original dataset"""
        augmented_data = []
        print(f"Augmenting dataset with factor: {augmentation_factor}")
        for user_data in original_dataset:
            for _ in range(augmentation_factor):
                augmented_grades = self.add_noise_to_grades(user_data['academic_grades'])
                augmented_hobbies = self.add_variation_to_hobbies(user_data['hobbies'])
                augmented_personality = self.augment_personality_scores(user_data['personality'])
                
                augmented_profile = {
                    'academic_grades': augmented_grades,
                    'hobbies': augmented_hobbies,
                    'personality': augmented_personality,
                    'career': user_data['career']
                }
                augmented_data.append(augmented_profile)
        
        total_augmented_samples = len(original_dataset) * augmentation_factor
        print(f"Generated a total of {total_augmented_samples} augmented samples.")
        return augmented_data

