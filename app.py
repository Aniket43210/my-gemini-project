"""
Career Prediction Web UI
========================

A simple Flask web application that provides a user interface
for the career prediction model.

Usage:
    python app.py
"""

from flask import Flask, render_template, request, jsonify
import joblib
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from main import create_ultimate_predictor

app = Flask(__name__)

# Global variable to store loaded models
models = None

def load_models():
    """Load all trained models"""
    global models
    try:
        models = {
            'broad_model': joblib.load('models/ultimate_broad_model.joblib'),
            'field_model': joblib.load('models/ultimate_field_model.joblib'),
            'career_model': joblib.load('models/ultimate_career_model.joblib'),
            'broad_encoder': joblib.load('models/broad_encoder.joblib'),
            'field_encoder': joblib.load('models/field_encoder.joblib'),
            'career_encoder': joblib.load('models/career_encoder.joblib')
        }
        print("✓ Models loaded successfully")
        return True
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please run 'python main.py' first to train the models.")
        return False
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def convert_grade_to_float(grade_str):
    """Convert letter grade to float value"""
    grade_map = {
        'A+': 1.0, 'A': 0.9, 'A-': 0.85,
        'B+': 0.8, 'B': 0.7, 'B-': 0.65,
        'C+': 0.6, 'C': 0.5, 'C-': 0.45,
        'D+': 0.4, 'D': 0.3, 'D-': 0.25,
        'F': 0.0
    }
    return grade_map.get(grade_str.upper(), 0.5)

@app.route('/')
def index():
    """Main page with the career prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle career prediction request"""
    if not models:
        return jsonify({
            'error': 'Models not loaded. Please ensure models are trained first.'
        }), 500
    
    try:
        # Get form data
        data = request.json
        
        # Process academic grades
        academic_grades = {
            'mathematics': convert_grade_to_float(data['academic']['mathematics']),
            'science': convert_grade_to_float(data['academic']['science']),
            'english': convert_grade_to_float(data['academic']['english']),
            'social_science': convert_grade_to_float(data['academic']['social_science']),
            'second_language': convert_grade_to_float(data['academic']['second_language'])
        }
        
        # Process hobbies
        hobbies = {}
        for hobby_name, hobby_data in data['hobbies'].items():
            if hobby_data['selected']:
                hobbies[hobby_name] = {
                    'intensity': float(hobby_data['intensity']) / 10.0,
                    'proficiency': float(hobby_data['proficiency']) / 10.0,
                    'years': int(hobby_data['years'])
                }
        
        # Process personality traits
        personality = {
            'openness': float(data['personality']['openness']) / 10.0,
            'conscientiousness': float(data['personality']['conscientiousness']) / 10.0,
            'extraversion': float(data['personality']['extraversion']) / 10.0,
            'agreeableness': float(data['personality']['agreeableness']) / 10.0,
            'neuroticism': float(data['personality']['neuroticism']) / 10.0
        }
        
        # Create predictor and make prediction
        predictor = create_ultimate_predictor(models)
        result = predictor.predict_user_career(academic_grades, hobbies, personality)
        
        # Format response
        response = {
            'success': True,
            'primary_recommendation': {
                'career': result['primary_recommendation']['career'],
                'confidence': f"{result['primary_recommendation']['confidence']:.1%}"
            },
            'hierarchical': {
                'broad': {
                    'category': result['hierarchical_predictions']['broad']['category'],
                    'confidence': f"{result['hierarchical_predictions']['broad']['confidence']:.1%}"
                },
                'field': {
                    'category': result['hierarchical_predictions']['field']['category'],
                    'confidence': f"{result['hierarchical_predictions']['field']['confidence']:.1%}"
                },
                'specific': {
                    'category': result['hierarchical_predictions']['specific']['category'],
                    'confidence': f"{result['hierarchical_predictions']['specific']['confidence']:.1%}"
                }
            },
            'alternatives': []
        }
        
        # Add top alternatives
        for alt in result['top_alternatives']['careers'][:5]:
            response['alternatives'].append({
                'career': alt['career'],
                'confidence': f"{alt['confidence']:.1%}"
            })
        
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
        'models_loaded': models is not None
    })

if __name__ == '__main__':
    import os
    print("🚀 Starting Career Prediction Web UI...")
    print("=" * 50)
    
    # Load models at startup
    if load_models():
        print("\n🌐 Starting web server...")
        port = int(os.environ.get('PORT', 5000))
        print(f"🔗 Server will start on port: {port}")
        print("🛑 Press Ctrl+C to stop the server")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("\n❌ Cannot start web server without trained models.")
        print("Please run 'python main.py' first to train the models.")
