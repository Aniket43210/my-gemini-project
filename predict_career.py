import joblib
from main import create_ultimate_predictor, create_ultimate_features

def predict_career_example():
    # Load trained models
    try:
        results = {
            'broad_model': joblib.load('models/ultimate_broad_model.joblib'),
            'field_model': joblib.load('models/ultimate_field_model.joblib'),
            'career_model': joblib.load('models/ultimate_career_model.joblib'),
            'broad_encoder': joblib.load('models/broad_encoder.joblib'),
            'field_encoder': joblib.load('models/field_encoder.joblib'),
            'career_encoder': joblib.load('models/career_encoder.joblib')
        }
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Make sure you have trained the models first by running main.py")
        return

    # Create predictor
    predictor = create_ultimate_predictor(results)

    # Make prediction
    try:
        prediction = predictor.predict_user_career(
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

        print("\n=== Career Prediction Results ===")
        print(f"Recommended Career: {prediction['primary_recommendation']['career']}")
        print(f"Confidence: {prediction['primary_recommendation']['confidence']:.1%}")

        if 'alternative_recommendations' in prediction:
            print("\nAlternative Careers:")
            for alt in prediction['alternative_recommendations'][:3]:
                print(f"- {alt['career']} (Confidence: {alt['confidence']:.1%})")

    except Exception as e:
        print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    predict_career_example()
