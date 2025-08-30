"""
Career Prediction Web App
Flask application for career prediction using pre-trained ML models
"""

import os
import sys
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

# Add src to path
sys.path.append('src')
sys.path.append('.')

from main import create_ultimate_features

app = Flask(__name__)

# Global variables for models
predictor = None
models_loaded = False

def load_models():
    """Load all pre-trained models"""
    global predictor, models_loaded
    
    try:
        model_dir = "models"
        
        # Check if models exist
        required_files = [
            f"{model_dir}/ultimate_broad_model.joblib",
            f"{model_dir}/ultimate_field_model.joblib", 
            f"{model_dir}/ultimate_career_model.joblib",
            f"{model_dir}/broad_encoder.joblib",
            f"{model_dir}/field_encoder.joblib",
            f"{model_dir}/career_encoder.joblib"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"Missing model files: {missing_files}")
            print("Run 'python main.py' to train models first")
            return False
        
        # Load models
        results = {
            'broad_model': joblib.load(f"{model_dir}/ultimate_broad_model.joblib"),
            'field_model': joblib.load(f"{model_dir}/ultimate_field_model.joblib"),
            'career_model': joblib.load(f"{model_dir}/ultimate_career_model.joblib"),
            'broad_encoder': joblib.load(f"{model_dir}/broad_encoder.joblib"),
            'field_encoder': joblib.load(f"{model_dir}/field_encoder.joblib"),
            'career_encoder': joblib.load(f"{model_dir}/career_encoder.joblib")
        }
        
        # Create predictor class
        class UltimateCareerPredictor:
            def __init__(self, results):
                self.broad_model = results['broad_model']
                self.field_model = results['field_model']
                self.career_model = results['career_model']
                self.broad_encoder = results['broad_encoder']
                self.field_encoder = results['field_encoder']
                self.career_encoder = results['career_encoder']
                
            def predict_user_career(self, academic_grades, hobbies, personality):
                """Make hierarchical predictions with confidence scoring"""
                # Create user features
                user_data = [{
                    'academic_grades': academic_grades,
                    'hobbies': hobbies,
                    'personality': personality,
                    'career': 'unknown'
                }]
                
                user_features, _ = create_ultimate_features(user_data)
                
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
                        f'Analysis based on {user_features.shape[1]} engineered features and hierarchical ensemble learning'
                    ]
                }
        
        predictor = UltimateCareerPredictor(results)
        models_loaded = True
        
        print("Models loaded successfully!")
        print(f"   Broad categories: {len(results['broad_encoder'].classes_)}")
        print(f"   Fields: {len(results['field_encoder'].classes_)}")
        print(f"   Careers: {len(results['career_encoder'].classes_)}")
        
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please run python main.py to train models first.'
            }), 500
        
        # Get form data
        data = request.get_json()
        
        # Extract academic grades
        academic_grades = {
            'mathematics': float(data.get('mathematics', 0.5)),
            'science': float(data.get('science', 0.5)),
            'english': float(data.get('english', 0.5)),
            'social_science': float(data.get('social_science', 0.5)),
            'second_language': float(data.get('second_language', 0.5))
        }
        
        # Extract personality traits
        personality = {
            'openness': float(data.get('openness', 0.5)),
            'conscientiousness': float(data.get('conscientiousness', 0.5)),
            'extraversion': float(data.get('extraversion', 0.5)),
            'agreeableness': float(data.get('agreeableness', 0.5)),
            'neuroticism': float(data.get('neuroticism', 0.5))
        }
        
        # Extract hobbies
        hobbies = {}
        hobby_names = ['programming', 'research', 'writing', 'music', 'sports', 
                      'cooking', 'volunteering', 'photography', 'gaming', 'robotics', 
                      'reading', 'entrepreneurship']
        
        for hobby in hobby_names:
            if data.get(f'hobby_{hobby}'):
                intensity = float(data.get(f'hobby_{hobby}_intensity', 0.5))
                proficiency = float(data.get(f'hobby_{hobby}_proficiency', 0.5))
                years = float(data.get(f'hobby_{hobby}_years', 1))
                
                hobbies[hobby] = {
                    'intensity': intensity,
                    'proficiency': proficiency,
                    'years': years
                }
        
        # Make prediction
        result = predictor.predict_user_career(academic_grades, hobbies, personality)
        
        # Format response
        response = {
            'success': True,
            'primary_prediction': {
                'career': result['primary_recommendation']['career'],
                'confidence': f"{result['primary_recommendation']['confidence']:.1%}"
            },
            'hierarchical': {
                'broad_category': {
                    'name': result['hierarchical_predictions']['broad']['category'],
                    'confidence': f"{result['hierarchical_predictions']['broad']['confidence']:.1%}"
                },
                'field': {
                    'name': result['hierarchical_predictions']['field']['category'],
                    'confidence': f"{result['hierarchical_predictions']['field']['confidence']:.1%}"
                },
                'specific': {
                    'name': result['hierarchical_predictions']['specific']['category'],
                    'confidence': f"{result['hierarchical_predictions']['specific']['confidence']:.1%}"
                }
            },
            'alternatives': {
                'careers': [
                    {
                        'name': alt['career'],
                        'confidence': f"{alt['confidence']:.1%}"
                    }
                    for alt in result['top_alternatives']['careers'][:5]
                ],
                'broad_categories': [
                    {
                        'name': alt['category'],
                        'confidence': f"{alt['confidence']:.1%}"
                    }
                    for alt in result['top_alternatives']['broad_categories'][:3]
                ],
                'fields': [
                    {
                        'name': alt['category'],
                        'confidence': f"{alt['confidence']:.1%}"
                    }
                    for alt in result['top_alternatives']['fields'][:3]
                ]
            },
            'reasoning': result['recommendation_reasoning']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

if __name__ == '__main__':
    print("Career Prediction Web App")
    print("="*40)
    
    # Load models on startup
    if load_models():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    else:
        print("Failed to load models. Please run 'python main.py' first.")
        sys.exit(1)
