import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.hobby_taxonomy import HOBBY_TAXONOMY, HOBBY_SYNERGIES, ACADEMIC_CAREER_WEIGHTS, CAREER_HIERARCHY

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.hobby_facet_names = self._get_all_hobby_facets()
        
    def _get_all_hobby_facets(self):
        """Extract all unique hobby facets for consistent feature naming"""
        facets = set()
        for hobby_data in HOBBY_TAXONOMY.values():
            facets.update(hobby_data['facets'].keys())
        return sorted(list(facets))
    
    def process_multifaceted_hobbies(self, user_hobbies):
        """
        Process hobbies with multiple facets and intensity levels
        
        Args:
            user_hobbies: dict with hobby names as keys and dict of attributes as values
                         e.g., {'programming': {'intensity': 0.8, 'proficiency': 0.7, 'years': 3}}
        """
        hobby_vectors = defaultdict(float)
        
        # Process each hobby with its facets
        for hobby_name, hobby_details in user_hobbies.items():
            if hobby_name not in HOBBY_TAXONOMY:
                continue
                
            intensity = hobby_details.get('intensity', 0.5)
            proficiency = hobby_details.get('proficiency', 0.5)
            years_experience = hobby_details.get('years', 1)
            
            # Calculate experience weight (diminishing returns after 5 years)
            experience_weight = min(years_experience / 5.0, 1.0)
            overall_weight = intensity * proficiency * experience_weight
            
            # Apply facet scores with weighting
            facets = HOBBY_TAXONOMY[hobby_name]['facets']
            for facet, base_score in facets.items():
                weighted_score = base_score * overall_weight
                hobby_vectors[facet] += weighted_score
        
        # Create feature vector with all possible facets
        feature_vector = {}
        for facet in self.hobby_facet_names:
            feature_vector[f'hobby_{facet}'] = hobby_vectors.get(facet, 0.0)
        
        return feature_vector
    
    def detect_hobby_synergies(self, user_hobbies):
        """Detect powerful hobby combinations that boost certain careers"""
        synergy_features = {}
        hobby_list = list(user_hobbies.keys())
        
        for i in range(len(hobby_list)):
            for j in range(i+1, len(hobby_list)):
                combo = tuple(sorted([hobby_list[i], hobby_list[j]]))
                
                if combo in HOBBY_SYNERGIES:
                    synergy_data = HOBBY_SYNERGIES[combo]
                    synergy_score = synergy_data['synergy_score']
                    
                    # Weight by both hobbies' intensities
                    hobby1_intensity = user_hobbies[hobby_list[i]].get('intensity', 0.5)
                    hobby2_intensity = user_hobbies[hobby_list[j]].get('intensity', 0.5)
                    
                    combined_intensity = (hobby1_intensity + hobby2_intensity) / 2
                    final_synergy = synergy_score * combined_intensity
                    
                    synergy_name = f"synergy_{combo[0]}_{combo[1]}"
                    synergy_features[synergy_name] = final_synergy
        
        return synergy_features
    
    def create_academic_profiles(self, academic_grades):
        """
        Create academic profile features based on subject performance patterns
        
        Args:
            academic_grades: dict with subject names and grades (0-1 scale)
        """
        features = {}
        
        # Direct subject scores
        subjects = ['mathematics', 'science', 'social_science', 'english', 'second_language']
        for subject in subjects:
            features[f'grade_{subject}'] = academic_grades.get(subject, 0.5)
        
        # Academic profile patterns
        stem_subjects = ['mathematics', 'science']
        humanities_subjects = ['english', 'social_science', 'second_language']
        
        stem_avg = np.mean([academic_grades.get(s, 0.5) for s in stem_subjects])
        humanities_avg = np.mean([academic_grades.get(s, 0.5) for s in humanities_subjects])
        
        features['stem_orientation'] = stem_avg
        features['humanities_orientation'] = humanities_avg
        features['stem_humanities_ratio'] = stem_avg / (humanities_avg + 1e-6)
        features['academic_consistency'] = 1 - np.std(list(academic_grades.values()))
        features['overall_gpa'] = np.mean(list(academic_grades.values()))
        
        # Subject combinations that predict specific career paths
        features['math_science_combo'] = academic_grades.get('mathematics', 0.5) * academic_grades.get('science', 0.5)
        features['language_social_combo'] = academic_grades.get('english', 0.5) * academic_grades.get('social_science', 0.5)
        
        return features
    
    def process_personality_features(self, personality_scores):
        """
        Process Big Five personality traits with derived features
        
        Args:
            personality_scores: dict with Big Five traits (0-1 scale)
        """
        features = {}
        
        # Direct personality scores
        big_five = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in big_five:
            features[f'personality_{trait}'] = personality_scores.get(trait, 0.5)
        
        # Derived personality combinations
        features['leadership_potential'] = (
            personality_scores.get('extraversion', 0.5) * 0.4 +
            personality_scores.get('conscientiousness', 0.5) * 0.3 +
            personality_scores.get('openness', 0.5) * 0.3
        )
        
        features['analytical_disposition'] = (
            personality_scores.get('openness', 0.5) * 0.4 +
            personality_scores.get('conscientiousness', 0.5) * 0.4 +
            (1 - personality_scores.get('extraversion', 0.5)) * 0.2
        )
        
        features['people_orientation'] = (
            personality_scores.get('extraversion', 0.5) * 0.4 +
            personality_scores.get('agreeableness', 0.5) * 0.4 +
            (1 - personality_scores.get('neuroticism', 0.5)) * 0.2
        )
        
        features['innovation_score'] = (
            personality_scores.get('openness', 0.5) * 0.6 +
            personality_scores.get('conscientiousness', 0.5) * 0.4
        )
        
        return features
    
    def create_interaction_features(self, academic_features, hobby_features, personality_features):
        """Create interaction features between different data types"""
        interactions = {}
        
        # Academic-Hobby interactions
        stem_score = academic_features.get('stem_orientation', 0.5)
        tech_hobby_score = hobby_features.get('hobby_technical_skills', 0.0)
        interactions['stem_tech_alignment'] = stem_score * tech_hobby_score
        
        # Academic-Personality interactions
        analytical_personality = personality_features.get('analytical_disposition', 0.5)
        math_grade = academic_features.get('grade_mathematics', 0.5)
        interactions['analytical_math_synergy'] = analytical_personality * math_grade
        
        # Hobby-Personality interactions
        creative_hobby = hobby_features.get('hobby_creative_arts', 0.0)
        openness = personality_features.get('personality_openness', 0.5)
        interactions['creative_openness_match'] = creative_hobby * openness
        
        leadership_potential = personality_features.get('leadership_potential', 0.5)
        social_hobbies = hobby_features.get('hobby_social_interaction', 0.0)
        interactions['leadership_social_alignment'] = leadership_potential * social_hobbies
        
        return interactions
    
    def create_diversity_features(self, user_data):
        """Create features that measure diversity and specialization"""
        features = {}
        
        # Hobby diversity
        hobby_scores = [v for k, v in user_data.items() if k.startswith('hobby_') and v > 0]
        if hobby_scores:
            features['hobby_diversity'] = len(hobby_scores) / len(self.hobby_facet_names)
            features['hobby_specialization'] = max(hobby_scores)
            features['hobby_breadth_depth_ratio'] = len(hobby_scores) / (max(hobby_scores) + 1e-6)
        else:
            features['hobby_diversity'] = 0
            features['hobby_specialization'] = 0
            features['hobby_breadth_depth_ratio'] = 0
        
        # Academic balance
        academic_scores = [v for k, v in user_data.items() if k.startswith('grade_')]
        if academic_scores:
            features['academic_balance'] = 1 - np.std(academic_scores)
            features['academic_peak_performance'] = max(academic_scores)
        else:
            features['academic_balance'] = 0.5
            features['academic_peak_performance'] = 0.5
        
        return features
    
    def engineer_all_features(self, academic_grades, user_hobbies, personality_scores):
        """
        Main method to create all engineered features
        
        Returns:
            dict: Complete feature vector ready for model training
        """
        # Process each data type
        academic_features = self.create_academic_profiles(academic_grades)
        hobby_features = self.process_multifaceted_hobbies(user_hobbies)
        synergy_features = self.detect_hobby_synergies(user_hobbies)
        personality_features = self.process_personality_features(personality_scores)
        
        # Combine all basic features
        all_features = {}
        all_features.update(academic_features)
        all_features.update(hobby_features)
        all_features.update(synergy_features)
        all_features.update(personality_features)
        
        # Create interaction features
        interaction_features = self.create_interaction_features(
            academic_features, hobby_features, personality_features
        )
        all_features.update(interaction_features)
        
        # Create diversity features
        diversity_features = self.create_diversity_features(all_features)
        all_features.update(diversity_features)
        
        return all_features
    
    def prepare_training_data(self, user_data_list):
        """
        Prepare multiple user records for training
        
        Args:
            user_data_list: List of dicts, each containing academic_grades, hobbies, personality, career
        
        Returns:
            X: Feature matrix, y: Career labels
        """
        feature_vectors = []
        career_labels = []
        
        for user_data in user_data_list:
            features = self.engineer_all_features(
                user_data['academic_grades'],
                user_data['hobbies'],
                user_data['personality']
            )
            feature_vectors.append(features)
            career_labels.append(user_data['career'])
        
        # Convert to DataFrame
        X = pd.DataFrame(feature_vectors)
        
        # Fill missing values with 0
        X = X.fillna(0)
        
        # Encode labels
        y = self.label_encoder.fit_transform(career_labels)
        
        return X, y
