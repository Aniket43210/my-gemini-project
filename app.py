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

from simple_predictor import SimpleCareerPredictor

app = Flask(__name__)

# Global variables for models
predictor = None
models_loaded = False

def load_models():
    """Load all pre-trained models"""
    global predictor, models_loaded
    
    try:
        # Create predictor
        predictor = SimpleCareerPredictor()
        models_loaded = predictor.models_loaded
        
        return models_loaded
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
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
