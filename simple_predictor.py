"""
Simple Career Predictor
Loads pre-trained models and provides prediction functionality without the full training pipeline
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')
sys.path.append('.')

def safe_float(value, default=0.5):
    """Safely convert value to float with fallback"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def create_prediction_features(academic_grades, hobbies, personality):
    """Create feature matrix for prediction"""
    
    feature_dict = {}
    
    # === ACADEMIC FEATURES (5 features) ===
    feature_dict['math_grade'] = safe_float(academic_grades.get('mathematics', 0.5))
    feature_dict['science_grade'] = safe_float(academic_grades.get('science', 0.5))
    feature_dict['english_grade'] = safe_float(academic_grades.get('english', 0.5))
    feature_dict['social_grade'] = safe_float(academic_grades.get('social_science', 0.5))
    feature_dict['language_grade'] = safe_float(academic_grades.get('second_language', 0.5))
    
    # === PERSONALITY FEATURES (5 features) ===
    feature_dict['openness'] = safe_float(personality.get('openness', 0.5))
    feature_dict['conscientiousness'] = safe_float(personality.get('conscientiousness', 0.5))
    feature_dict['extraversion'] = safe_float(personality.get('extraversion', 0.5))
    feature_dict['agreeableness'] = safe_float(personality.get('agreeableness', 0.5))
    feature_dict['neuroticism'] = safe_float(personality.get('neuroticism', 0.5))
    
    # === HOBBY FEATURES ===
    feature_dict['num_hobbies'] = len(hobbies)
    
    # Calculate hobby statistics safely
    if hobbies:
        intensities = []
        proficiencies = []
        years_list = []
        
        for hobby_data in hobbies.values():
            if isinstance(hobby_data, dict):
                intensities.append(safe_float(hobby_data.get('intensity', 0.5)))
                proficiencies.append(safe_float(hobby_data.get('proficiency', 0.5)))
                years_list.append(safe_float(hobby_data.get('years', 1)))
        
        feature_dict['avg_hobby_intensity'] = np.mean(intensities) if intensities else 0
        feature_dict['avg_hobby_proficiency'] = np.mean(proficiencies) if proficiencies else 0
        feature_dict['avg_hobby_years'] = np.mean(years_list) if years_list else 0
        feature_dict['max_hobby_intensity'] = np.max(intensities) if intensities else 0
        feature_dict['max_hobby_proficiency'] = np.max(proficiencies) if proficiencies else 0
        feature_dict['hobby_intensity_std'] = np.std(intensities) if len(intensities) > 1 else 0
        feature_dict['hobby_proficiency_std'] = np.std(proficiencies) if len(proficiencies) > 1 else 0
    else:
        feature_dict.update({
            'avg_hobby_intensity': 0, 'avg_hobby_proficiency': 0, 'avg_hobby_years': 0,
            'max_hobby_intensity': 0, 'max_hobby_proficiency': 0,
            'hobby_intensity_std': 0, 'hobby_proficiency_std': 0
        })
    
    # === BINARY HOBBY FEATURES (12 features) ===
    hobby_names = list(hobbies.keys())
    feature_dict['has_programming'] = 1 if 'programming' in hobby_names else 0
    feature_dict['has_research'] = 1 if 'research' in hobby_names else 0
    feature_dict['has_writing'] = 1 if 'writing' in hobby_names else 0
    feature_dict['has_music'] = 1 if 'music' in hobby_names else 0
    feature_dict['has_sports'] = 1 if any('sport' in h.lower() for h in hobby_names) else 0
    feature_dict['has_cooking'] = 1 if 'cooking' in hobby_names else 0
    feature_dict['has_volunteering'] = 1 if 'volunteering' in hobby_names else 0
    feature_dict['has_photography'] = 1 if 'photography' in hobby_names else 0
    feature_dict['has_gaming'] = 1 if 'gaming' in hobby_names else 0
    feature_dict['has_robotics'] = 1 if 'robotics' in hobby_names else 0
    feature_dict['has_reading'] = 1 if 'reading' in hobby_names else 0
    feature_dict['has_entrepreneurship'] = 1 if 'entrepreneurship' in hobby_names else 0
    
    # === DERIVED ACADEMIC FEATURES (6 features) ===
    feature_dict['stem_score'] = (feature_dict['math_grade'] + feature_dict['science_grade']) / 2
    feature_dict['humanities_score'] = (feature_dict['english_grade'] + feature_dict['social_grade']) / 2
    feature_dict['stem_vs_humanities'] = feature_dict['stem_score'] - feature_dict['humanities_score']
    feature_dict['academic_consistency'] = 1 - np.std([
        feature_dict['math_grade'], feature_dict['science_grade'], 
        feature_dict['english_grade'], feature_dict['social_grade'], 
        feature_dict['language_grade']
    ])
    feature_dict['academic_peak'] = max([
        feature_dict['math_grade'], feature_dict['science_grade'], 
        feature_dict['english_grade'], feature_dict['social_grade'], 
        feature_dict['language_grade']
    ])
    feature_dict['academic_average'] = np.mean([
        feature_dict['math_grade'], feature_dict['science_grade'], 
        feature_dict['english_grade'], feature_dict['social_grade'], 
        feature_dict['language_grade']
    ])
    
    # === ADVANCED PERSONALITY-DERIVED FEATURES (8 features) ===
    feature_dict['leadership_potential'] = (
        feature_dict['extraversion'] * 0.4 + 
        feature_dict['conscientiousness'] * 0.3 + 
        feature_dict['openness'] * 0.3
    )
    
    feature_dict['analytical_disposition'] = (
        feature_dict['openness'] * 0.4 +
        feature_dict['conscientiousness'] * 0.4 +
        (1 - feature_dict['extraversion']) * 0.2
    )
    
    feature_dict['people_orientation'] = (
        feature_dict['extraversion'] * 0.4 +
        feature_dict['agreeableness'] * 0.4 +
        (1 - feature_dict['neuroticism']) * 0.2
    )
    
    feature_dict['stress_resilience'] = (
        (1 - feature_dict['neuroticism']) * 0.5 +
        feature_dict['conscientiousness'] * 0.3 +
        feature_dict['extraversion'] * 0.2
    )
    
    feature_dict['innovation_potential'] = (
        feature_dict['openness'] * 0.6 +
        feature_dict['extraversion'] * 0.2 +
        (1 - feature_dict['neuroticism']) * 0.2
    )
    
    feature_dict['detail_orientation'] = (
        feature_dict['conscientiousness'] * 0.6 +
        (1 - feature_dict['neuroticism']) * 0.4
    )
    
    feature_dict['collaboration_score'] = (
        feature_dict['agreeableness'] * 0.5 +
        feature_dict['extraversion'] * 0.3 +
        feature_dict['conscientiousness'] * 0.2
    )
    
    feature_dict['adaptability_score'] = (
        feature_dict['openness'] * 0.5 +
        (1 - feature_dict['neuroticism']) * 0.3 +
        feature_dict['extraversion'] * 0.2
    )
    
    # === ORIENTATION FEATURES (4 features) ===
    feature_dict['technical_orientation'] = (
        feature_dict['stem_score'] * 0.6 + 
        feature_dict['has_programming'] * 0.4
    )
    
    creative_hobbies = (feature_dict['has_music'] + feature_dict['has_photography'] + 
                      feature_dict['has_writing'] + feature_dict['has_cooking']) / 4
    feature_dict['creative_orientation'] = (
        creative_hobbies * 0.6 +
        feature_dict['openness'] * 0.4
    )
    
    social_hobbies = (feature_dict['has_volunteering'] + feature_dict['has_sports']) / 2
    feature_dict['social_orientation'] = (
        social_hobbies * 0.5 + 
        feature_dict['people_orientation'] * 0.5
    )
    
    research_orientation = (feature_dict['has_research'] + feature_dict['has_reading']) / 2
    feature_dict['research_orientation'] = (
        research_orientation * 0.6 +
        feature_dict['analytical_disposition'] * 0.4
    )
    
    # === SPECIALIZATION INDICES (3 features) ===
    feature_dict['hobby_specialization_index'] = (
        feature_dict['max_hobby_intensity'] / (feature_dict['avg_hobby_intensity'] + 0.001)
    )
    
    feature_dict['academic_specialization_index'] = (
        feature_dict['academic_peak'] / (feature_dict['academic_average'] + 0.001)
    )
    
    personality_scores = [feature_dict['openness'], feature_dict['conscientiousness'], 
                        feature_dict['extraversion'], feature_dict['agreeableness'], 
                        feature_dict['neuroticism']]
    feature_dict['personality_extremity'] = np.std(personality_scores)
    
    # Convert to DataFrame
    X = pd.DataFrame([feature_dict])
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X

class SimpleCareerPredictor:
    """Simple career predictor using pre-trained models"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models_loaded = False
        self.demo_mode = False
        self.load_models()
    
    def load_models(self):
        """Load all pre-trained models"""
        try:
            required_files = [
                f"{self.model_dir}/ultimate_broad_model.joblib",
                f"{self.model_dir}/ultimate_field_model.joblib", 
                f"{self.model_dir}/ultimate_career_model.joblib",
                f"{self.model_dir}/broad_encoder.joblib",
                f"{self.model_dir}/field_encoder.joblib",
                f"{self.model_dir}/career_encoder.joblib"
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                print(f"Missing model files: {missing_files}")
                print("Run 'python main.py' to train models first")
                return False
            
            # Load models
            self.broad_model = joblib.load(f"{self.model_dir}/ultimate_broad_model.joblib")
            self.field_model = joblib.load(f"{self.model_dir}/ultimate_field_model.joblib")
            self.career_model = joblib.load(f"{self.model_dir}/ultimate_career_model.joblib")
            self.broad_encoder = joblib.load(f"{self.model_dir}/broad_encoder.joblib")
            self.field_encoder = joblib.load(f"{self.model_dir}/field_encoder.joblib")
            self.career_encoder = joblib.load(f"{self.model_dir}/career_encoder.joblib")
            
            self.models_loaded = True
            print("Models loaded successfully!")
            print(f"   Broad categories: {len(self.broad_encoder.classes_)}")
            print(f"   Fields: {len(self.field_encoder.classes_)}")
            print(f"   Careers: {len(self.career_encoder.classes_)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Switching to demo mode with mock predictions")
            self.demo_mode = True
            self.models_loaded = True
            return True
    
    def predict_user_career(self, academic_grades, hobbies, personality):
        """Make hierarchical predictions with confidence scoring"""
        if not self.models_loaded:
            raise Exception("Models not loaded")
        
        if self.demo_mode:
            return self._demo_prediction(academic_grades, hobbies, personality)
        
        # Create user features
        user_features = create_prediction_features(academic_grades, hobbies, personality)
        
        # Make broad category prediction
        broad_proba = self.broad_model.predict_proba(user_features)[0]
        broad_pred = self.broad_encoder.inverse_transform([np.argmax(broad_proba)])[0]
        broad_confidence = max(broad_proba)
        
        # Make field prediction
        field_proba = self.field_model.predict_proba(user_features)[0]
        field_pred = self.field_encoder.inverse_transform([np.argmax(field_proba)])[0]
        field_confidence = max(field_proba)
        
        # Make career prediction
        career_proba = self.career_model.predict_proba(user_features)[0]
        career_pred = self.career_encoder.inverse_transform([np.argmax(career_proba)])[0]
        career_confidence = max(career_proba)
        
        # Get top alternatives for each level
        top_broad_indices = np.argsort(broad_proba)[-3:][::-1]
        top_broad_alternatives = [
            {'category': self.broad_encoder.inverse_transform([idx])[0], 'confidence': broad_proba[idx]}
            for idx in top_broad_indices
        ]
        
        top_field_indices = np.argsort(field_proba)[-3:][::-1]
        top_field_alternatives = [
            {'category': self.field_encoder.inverse_transform([idx])[0], 'confidence': field_proba[idx]}
            for idx in top_field_indices
        ]
        
        top_career_indices = np.argsort(career_proba)[-5:][::-1]
        top_career_alternatives = [
            {'career': self.career_encoder.inverse_transform([idx])[0], 'confidence': career_proba[idx]}
            for idx in top_career_indices
        ]
        
        return {
            'primary_recommendation': {
                'career': career_pred,
                'confidence': career_confidence,
                'level': 'specific'
            },
            'hierarchical_predictions': {
                'broad': {'category': broad_pred, 'confidence': broad_confidence},
                'field': {'category': field_pred, 'confidence': field_confidence},
                'specific': {'category': career_pred, 'confidence': career_confidence}
            },
            'top_alternatives': {
                'broad_categories': top_broad_alternatives,
                'fields': top_field_alternatives,
                'careers': top_career_alternatives
            },
            'recommendation_reasoning': [
                f'Broad category match: {broad_pred} (confidence: {broad_confidence:.1%})',
                f'Field specialization: {field_pred} (confidence: {field_confidence:.1%})',
                f'Specific career recommendation: {career_pred} (confidence: {career_confidence:.1%})',
                f'Analysis based on 51 engineered features and hierarchical ensemble learning'
            ]
        }
    
    def _demo_prediction(self, academic_grades, hobbies, personality):
        """Generate demo predictions based on user input patterns"""
        # Create features for analysis
        user_features = create_prediction_features(academic_grades, hobbies, personality)
        features = user_features.iloc[0]
        
        # Rule-based career prediction for demo
        stem_score = features['stem_score']
        humanities_score = features['humanities_score']
        has_programming = features['has_programming']
        has_research = features['has_research']
        has_writing = features['has_writing']
        has_music = features['has_music']
        has_entrepreneurship = features['has_entrepreneurship']
        extraversion = features['extraversion']
        openness = features['openness']
        
        # Determine broad category
        if stem_score > 0.6 and (has_programming or has_research):
            broad_category = 'STEM'
            broad_confidence = 0.85
        elif humanities_score > 0.6 and (has_writing or extraversion > 0.7):
            broad_category = 'Business'
            broad_confidence = 0.78
        elif openness > 0.7 and (has_music or has_writing):
            broad_category = 'Creative'
            broad_confidence = 0.82
        elif has_research or (stem_score > 0.5 and humanities_score > 0.5):
            broad_category = 'Healthcare'
            broad_confidence = 0.75
        else:
            broad_category = 'STEM'
            broad_confidence = 0.68
        
        # Determine field based on broad category
        if broad_category == 'STEM':
            if has_programming:
                field = 'Software Engineering'
                career = 'Software Engineer'
                field_confidence = 0.88
                career_confidence = 0.85
            elif has_research:
                field = 'Research/Science'
                career = 'Data Scientist' if stem_score > 0.7 else 'Research Scientist'
                field_confidence = 0.82
                career_confidence = 0.78
            else:
                field = 'Engineering'
                career = 'Systems Architect'
                field_confidence = 0.75
                career_confidence = 0.72
        elif broad_category == 'Business':
            if has_entrepreneurship:
                field = 'Business/Finance'
                career = 'Product Manager'
                field_confidence = 0.82
                career_confidence = 0.79
            elif extraversion > 0.7:
                field = 'Marketing/Sales'
                career = 'Marketing Manager'
                field_confidence = 0.85
                career_confidence = 0.81
            else:
                field = 'Business/Finance'
                career = 'Business Analyst'
                field_confidence = 0.78
                career_confidence = 0.74
        elif broad_category == 'Creative':
            if has_music:
                field = 'Design/Art'
                career = 'Art Director'
                field_confidence = 0.83
                career_confidence = 0.80
            elif has_writing:
                field = 'Design/Art'
                career = 'Content Creator'
                field_confidence = 0.81
                career_confidence = 0.77
            else:
                field = 'Design/Art'
                career = 'UX Designer'
                field_confidence = 0.79
                career_confidence = 0.76
        else:  # Healthcare
            field = 'Healthcare/Medical'
            career = 'Healthcare Analyst'
            field_confidence = 0.77
            career_confidence = 0.73
        
        # Generate alternatives
        careers_list = ['Software Engineer', 'Data Scientist', 'Product Manager', 'UX Designer', 
                       'Marketing Manager', 'Research Scientist', 'Business Analyst']
        fields_list = ['Software Engineering', 'Data Science/Analytics', 'Business/Finance', 
                      'Design/Art', 'Marketing/Sales']
        broad_list = ['STEM', 'Business', 'Creative', 'Healthcare']
        
        # Remove primary predictions from alternatives
        alt_careers = [c for c in careers_list if c != career]
        alt_fields = [f for f in fields_list if f != field]
        alt_broad = [b for b in broad_list if b != broad_category]
        
        return {
            'primary_recommendation': {
                'career': career,
                'confidence': career_confidence,
                'level': 'specific'
            },
            'hierarchical_predictions': {
                'broad': {'category': broad_category, 'confidence': broad_confidence},
                'field': {'category': field, 'confidence': field_confidence},
                'specific': {'category': career, 'confidence': career_confidence}
            },
            'top_alternatives': {
                'broad_categories': [
                    {'category': alt_broad[0], 'confidence': 0.65},
                    {'category': alt_broad[1], 'confidence': 0.58},
                    {'category': alt_broad[2], 'confidence': 0.42}
                ],
                'fields': [
                    {'category': alt_fields[0], 'confidence': 0.68},
                    {'category': alt_fields[1], 'confidence': 0.61},
                    {'category': alt_fields[2], 'confidence': 0.54}
                ],
                'careers': [
                    {'career': alt_careers[0], 'confidence': 0.71},
                    {'career': alt_careers[1], 'confidence': 0.66},
                    {'career': alt_careers[2], 'confidence': 0.59},
                    {'career': alt_careers[3], 'confidence': 0.52},
                    {'career': alt_careers[4], 'confidence': 0.47}
                ]
            },
            'recommendation_reasoning': [
                f'Broad category match: {broad_category} (confidence: {broad_confidence:.1%})',
                f'Field specialization: {field} (confidence: {field_confidence:.1%})',
                f'Specific career recommendation: {career} (confidence: {career_confidence:.1%})',
                f'Demo mode: Analysis based on rule-based matching and user profile patterns'
            ]
        }

if __name__ == '__main__':
    # Test the predictor
    predictor = SimpleCareerPredictor()
    
    if predictor.models_loaded:
        # Test prediction
        test_result = predictor.predict_user_career(
            academic_grades={
                'mathematics': 0.85, 'science': 0.80, 'english': 0.75,
                'social_science': 0.60, 'second_language': 0.65
            },
            hobbies={
                'programming': {'intensity': 0.9, 'proficiency': 0.8, 'years': 4}
            },
            personality={
                'openness': 0.85, 'conscientiousness': 0.75, 'extraversion': 0.45,
                'agreeableness': 0.65, 'neuroticism': 0.35
            }
        )
        
        print("\n=== TEST PREDICTION ===")
        print(f"Primary Career: {test_result['primary_recommendation']['career']}")
        print(f"Confidence: {test_result['primary_recommendation']['confidence']:.1%}")
        print("\nHierarchical breakdown:")
        for level, pred in test_result['hierarchical_predictions'].items():
            print(f"  {level.capitalize()}: {pred['category']} ({pred['confidence']:.1%})")
        
        print("\nAlternative careers:")
        for i, alt in enumerate(test_result['top_alternatives']['careers'][:3], 1):
            print(f"  {i}. {alt['career']} ({alt['confidence']:.1%})")
