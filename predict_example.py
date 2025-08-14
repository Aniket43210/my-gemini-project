"""
Career Prediction Example - User Input Requirements
==================================================

This script shows exactly what parameters you need from a user
to predict their career path using the trained model.

Usage:
    python predict_example.py
"""

import joblib
import sys
sys.path.append('.')

from main import create_ultimate_predictor

def load_trained_models():
    """Load all trained models"""
    try:
        results = {
            'broad_model': joblib.load('models/ultimate_broad_model.joblib'),
            'field_model': joblib.load('models/ultimate_field_model.joblib'),
            'career_model': joblib.load('models/ultimate_career_model.joblib'),
            'broad_encoder': joblib.load('models/broad_encoder.joblib'),
            'field_encoder': joblib.load('models/field_encoder.joblib'),
            'career_encoder': joblib.load('models/career_encoder.joblib')
        }
        return results
    except FileNotFoundError:
        print("❌ Models not found. Please run 'python main.py' first to train the models.")
        return None

def predict_career_example():
    """Example showing required user input parameters"""
    
    print("🎯 CAREER PREDICTION - USER INPUT REQUIREMENTS")
    print("=" * 60)
    
    # Load trained models
    results = load_trained_models()
    if not results:
        return
    
    predictor = create_ultimate_predictor(results)
    
    print("\n📋 REQUIRED INPUT PARAMETERS:")
    print("=" * 40)
    
    # ===== 1. ACADEMIC GRADES =====
    print("\n1️⃣ ACADEMIC GRADES (scale 0.0-1.0)")
    print("   • 0.0 = F grade, 0.5 = C grade, 1.0 = A+ grade")
    
    academic_grades = {
        'mathematics': 0.85,        # Math grade
        'science': 0.80,           # Science grade  
        'english': 0.75,           # English grade
        'social_science': 0.60,    # Social studies grade
        'second_language': 0.65    # Foreign language grade
    }
    
    print("   Required subjects:")
    for subject, grade in academic_grades.items():
        print(f"   • {subject}: {grade} (equivalent to {grade_to_letter(grade)} grade)")
    
    # ===== 2. HOBBIES =====
    print("\n2️⃣ HOBBIES (with intensity, proficiency, years)")
    print("   • intensity: How passionate about it (0.0-1.0)")
    print("   • proficiency: Skill level (0.0-1.0)")  
    print("   • years: Years of experience (integer)")
    
    hobbies = {
        'programming': {
            'intensity': 0.9,      # Very passionate
            'proficiency': 0.8,    # High skill level
            'years': 4             # 4 years experience
        },
        'photography': {
            'intensity': 0.7,      # Moderately passionate
            'proficiency': 0.6,    # Medium skill level
            'years': 2             # 2 years experience
        },
        'music': {
            'intensity': 0.6,
            'proficiency': 0.5,
            'years': 3
        }
    }
    
    print("   Available hobby categories:")
    available_hobbies = [
        'programming', 'research', 'writing', 'music', 'photography', 
        'cooking', 'volunteering', 'gaming', 'robotics', 'reading',
        'entrepreneurship', 'team_sports', 'individual_sports'
    ]
    
    for i, hobby in enumerate(available_hobbies, 1):
        print(f"   • {hobby}")
        if i % 4 == 0:  # New line every 4 items
            print()
    
    print("\n   Example hobbies input:")
    for hobby_name, details in hobbies.items():
        print(f"   • {hobby_name}: intensity={details['intensity']}, "
              f"proficiency={details['proficiency']}, years={details['years']}")
    
    # ===== 3. PERSONALITY TRAITS =====
    print("\n3️⃣ PERSONALITY TRAITS (Big Five, scale 0.0-1.0)")
    
    personality = {
        'openness': 0.85,          # Open to new experiences
        'conscientiousness': 0.75, # Organized, disciplined  
        'extraversion': 0.45,      # Social, outgoing
        'agreeableness': 0.65,     # Cooperative, trusting
        'neuroticism': 0.35        # Emotional instability (lower = more stable)
    }
    
    personality_descriptions = {
        'openness': 'Open to new experiences, creative, curious',
        'conscientiousness': 'Organized, disciplined, responsible',
        'extraversion': 'Social, outgoing, energetic',
        'agreeableness': 'Cooperative, trusting, helpful',
        'neuroticism': 'Emotional instability (lower = more stable)'
    }
    
    print("   Required personality traits:")
    for trait, score in personality.items():
        desc = personality_descriptions[trait]
        level = get_personality_level(score)
        print(f"   • {trait}: {score} ({level}) - {desc}")
    
    # ===== MAKE PREDICTION =====
    print("\n🔮 MAKING PREDICTION...")
    print("=" * 40)
    
    try:
        result = predictor.predict_user_career(academic_grades, hobbies, personality)
        
        # Display results
        primary = result['primary_recommendation']
        hierarchical = result['hierarchical_predictions']
        
        print(f"\n🎯 PRIMARY RECOMMENDATION: {primary['career']}")
        print(f"   Overall Confidence: {primary['confidence']:.1%}")
        
        print(f"\n📊 HIERARCHICAL BREAKDOWN:")
        print(f"   🌐 Broad Category: {hierarchical['broad']['category']} ({hierarchical['broad']['confidence']:.1%})")
        print(f"   🏢 Field Level: {hierarchical['field']['category']} ({hierarchical['field']['confidence']:.1%})")
        print(f"   🎯 Specific Career: {hierarchical['specific']['category']} ({hierarchical['specific']['confidence']:.1%})")
        
        print(f"\n🔍 TOP CAREER ALTERNATIVES:")
        for i, alt in enumerate(result['top_alternatives']['careers'][:3], 1):
            print(f"   {i}. {alt['career']} ({alt['confidence']:.1%})")
            
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return
    
    # ===== DATA COLLECTION GUIDE =====
    print(f"\n📝 DATA COLLECTION GUIDE FOR USERS:")
    print("=" * 50)
    print("""
    ACADEMIC GRADES:
    • Ask for grades in 5 core subjects
    • Convert to 0.0-1.0 scale (F=0.0, D=0.2, C=0.5, B=0.7, A=0.9, A+=1.0)
    
    HOBBIES:
    • Ask user to list their main hobbies/interests
    • For each hobby, ask:
      - How passionate are you? (1-10 scale, convert to 0.0-1.0)
      - How skilled are you? (1-10 scale, convert to 0.0-1.0)  
      - How many years have you been doing this?
    
    PERSONALITY TRAITS:
    • Use a personality questionnaire or ask user to self-assess:
      - How open are you to new experiences? (1-10)
      - How organized and disciplined are you? (1-10)
      - How social and outgoing are you? (1-10)
      - How cooperative and trusting are you? (1-10)
      - How emotionally stable are you? (1-10, then invert for neuroticism)
    • Convert all to 0.0-1.0 scale
    """)

def grade_to_letter(grade):
    """Convert numerical grade to letter grade"""
    if grade >= 0.95: return "A+"
    elif grade >= 0.9: return "A"
    elif grade >= 0.8: return "B+"
    elif grade >= 0.7: return "B"
    elif grade >= 0.6: return "C+"
    elif grade >= 0.5: return "C"
    elif grade >= 0.4: return "D+"
    elif grade >= 0.3: return "D"
    else: return "F"

def get_personality_level(score):
    """Convert personality score to descriptive level"""
    if score >= 0.8: return "Very High"
    elif score >= 0.6: return "High"
    elif score >= 0.4: return "Medium"
    elif score >= 0.2: return "Low"
    else: return "Very Low"

if __name__ == "__main__":
    predict_career_example()
