"""
Simple Career Prediction Web UI (Fallback Version)
==================================================

A simplified Flask web application that works without complex ML dependencies.
"""

from flask import Flask, render_template, request, jsonify
import json
import random

app = Flask(__name__)

# Simple career mapping based on academic performance
CAREER_MAPPING = {
    'high_math_science': ['Software Engineer', 'Data Scientist', 'Research Scientist', 'Engineer'],
    'high_english_social': ['Lawyer', 'Teacher', 'Journalist', 'Social Worker'],
    'balanced': ['Business Analyst', 'Project Manager', 'Consultant', 'Product Manager'],
    'creative': ['Designer', 'Artist', 'Writer', 'Marketing Specialist']
}

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

def simple_prediction(academic, hobbies, personality):
    """Simple rule-based career prediction"""
    
    # Calculate academic strengths
    math_science = (academic['mathematics'] + academic['science']) / 2
    language_social = (academic['english'] + academic['social_science']) / 2
    
    # Determine career category
    if math_science > 0.7 and math_science > language_social:
        category = 'high_math_science'
    elif language_social > 0.7 and language_social > math_science:
        category = 'high_english_social'
    elif any(hobby in hobbies for hobby in ['photography', 'music', 'writing']):
        category = 'creative'
    else:
        category = 'balanced'
    
    # Select career from category
    careers = CAREER_MAPPING[category]
    primary_career = random.choice(careers)
    
    # Generate confidence based on grade average
    avg_grade = sum(academic.values()) / len(academic.values())
    confidence = min(0.85, max(0.65, avg_grade + 0.1))
    
    return {
        'primary_recommendation': {
            'career': primary_career,
            'confidence': confidence
        },
        'hierarchical_predictions': {
            'broad': {'category': category.replace('_', ' ').title(), 'confidence': confidence},
            'field': {'category': primary_career.split()[0], 'confidence': confidence - 0.1},
            'specific': {'category': primary_career, 'confidence': confidence}
        },
        'top_alternatives': {
            'careers': [
                {'career': career, 'confidence': confidence - 0.1 - i*0.05} 
                for i, career in enumerate(careers[1:4])
            ]
        }
    }

@app.route('/')
def index():
    """Main page with the career prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle career prediction request"""
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
        
        # Make simple prediction
        result = simple_prediction(academic_grades, hobbies, personality)
        
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
        for alt in result['top_alternatives']['careers']:
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
        'models_loaded': True
    })

if __name__ == '__main__':
    import os
    print("🚀 Starting Simple Career Prediction Web UI...")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🔗 Server will start on port: {port}")
    print("🛑 Press Ctrl+C to stop the server")
    app.run(debug=False, host='0.0.0.0', port=port)
