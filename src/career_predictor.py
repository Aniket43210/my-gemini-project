import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import shap
import optuna
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import AdvancedFeatureEngineer
from config.hobby_taxonomy import CAREER_HIERARCHY

def create_career_to_field_mapping():
    """Create mapping from specific careers to field categories"""
    career_to_field = {}
    
    # Iterate through broad categories and their fields
    for broad_cat, fields in CAREER_HIERARCHY['broad_categories'].items():
        for field in fields:
            # Find all specific careers that belong to this broad category
            for career, career_broad in CAREER_HIERARCHY['specific_careers'].items():
                if career_broad == broad_cat:
                    # Map career to appropriate field based on career type
                    if 'engineer' in career or 'developer' in career:
                        career_to_field[career] = 'engineering'
                    elif 'data' in career or 'analyst' in career or 'scientist' in career:
                        career_to_field[career] = 'data_science' if 'data_science' in fields else 'research'
                    elif 'financial' in career or 'accountant' in career:
                        career_to_field[career] = 'finance'
                    elif 'marketing' in career:
                        career_to_field[career] = 'marketing'
                    elif 'manager' in career:
                        career_to_field[career] = 'management'
                    elif 'designer' in career:
                        career_to_field[career] = 'design'
                    elif 'chef' in career:
                        career_to_field[career] = 'arts'
                    elif 'teacher' in career:
                        career_to_field[career] = 'education'
                    elif 'psychologist' in career:
                        career_to_field[career] = 'counseling'
                    elif 'doctor' in career:
                        career_to_field[career] = 'medicine'
                    elif 'nurse' in career:
                        career_to_field[career] = 'nursing'
                    elif 'lawyer' in career:
                        career_to_field[career] = 'legal'
                    else:
                        # Default to first field in the broad category
                        career_to_field[career] = fields[0] if fields else broad_cat.lower()
    
    return career_to_field

# Create the global mapping
CAREER_TO_FIELD_MAPPING = create_career_to_field_mapping()

class HierarchicalCareerPredictor:
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.broad_model = None
        self.field_model = None
        self.specific_model = None
        self.label_encoders = {}
        self.confidence_thresholds = {
            'broad': 0.5,
            'field': 0.6,
            'specific': 0.65  # Lowered from 0.8 to be less conservative
        }
        
    def create_hierarchical_labels(self, career_labels):
        """Create hierarchical labels for broad, field, and specific categories"""
        broad_labels = []
        field_labels = []
        specific_labels = career_labels.copy()
        
        for career in career_labels:
            # Find broad category
            broad_category = CAREER_HIERARCHY.get('specific_careers', {}).get(career, 'Other')
            broad_labels.append(broad_category)
            
            # Find field category using the mapping
            field_category = CAREER_TO_FIELD_MAPPING.get(career, 'other')
            field_labels.append(field_category)
        
        return broad_labels, field_labels, specific_labels
    
    def optimize_hyperparameters(self, X, y, model_type='specific'):
        """Use Optuna for hyperparameter optimization"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def train_hierarchical_models(self, X, y_specific):
        """Train models at different hierarchy levels"""
        print("Creating hierarchical labels...")
        y_broad, y_field, y_specific = self.create_hierarchical_labels(y_specific)
        
        # Encode labels
        self.label_encoders['broad'] = LabelEncoder()
        self.label_encoders['field'] = LabelEncoder()
        self.label_encoders['specific'] = LabelEncoder()
        
        y_broad_encoded = self.label_encoders['broad'].fit_transform(y_broad)
        y_field_encoded = self.label_encoders['field'].fit_transform(y_field)
        y_specific_encoded = self.label_encoders['specific'].fit_transform(y_specific)
        
        # Split data
        X_train, X_test, y_broad_train, y_broad_test = train_test_split(
            X, y_broad_encoded, test_size=0.2, random_state=42, stratify=y_broad_encoded
        )
        _, _, y_field_train, y_field_test = train_test_split(
            X, y_field_encoded, test_size=0.2, random_state=42, stratify=y_field_encoded
        )
        _, _, y_specific_train, y_specific_test = train_test_split(
            X, y_specific_encoded, test_size=0.2, random_state=42, stratify=y_specific_encoded
        )
        
        # Train broad category model
        print("Training broad category model...")
        broad_params = self.optimize_hyperparameters(X_train, y_broad_train, 'broad')
        self.broad_model = xgb.XGBClassifier(**broad_params)
        self.broad_model.fit(X_train, y_broad_train)
        
        # Train field model
        print("Training field model...")
        field_params = self.optimize_hyperparameters(X_train, y_field_train, 'field')
        self.field_model = xgb.XGBClassifier(**field_params)
        self.field_model.fit(X_train, y_field_train)
        
        # Train specific model
        print("Training specific career model...")
        specific_params = self.optimize_hyperparameters(X_train, y_specific_train, 'specific')
        self.specific_model = xgb.XGBClassifier(**specific_params)
        self.specific_model.fit(X_train, y_specific_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_broad_test, y_field_test, y_specific_test)
        
        return self
    
    def evaluate_models(self, X_test, y_broad_test, y_field_test, y_specific_test):
        """Evaluate all hierarchical models"""
        print("\n=== MODEL EVALUATION ===")
        
        # Broad model evaluation
        broad_preds = self.broad_model.predict(X_test)
        broad_accuracy = accuracy_score(y_broad_test, broad_preds)
        print(f"Broad Category Accuracy: {broad_accuracy:.3f}")
        
        # Field model evaluation
        field_preds = self.field_model.predict(X_test)
        field_accuracy = accuracy_score(y_field_test, field_preds)
        print(f"Field Level Accuracy: {field_accuracy:.3f}")
        
        # Specific model evaluation
        specific_preds = self.specific_model.predict(X_test)
        specific_accuracy = accuracy_score(y_specific_test, specific_preds)
        print(f"Specific Career Accuracy: {specific_accuracy:.3f}")
        
        # Detailed classification report for specific model
        print("\nSpecific Career Classification Report:")
        specific_names = self.label_encoders['specific'].classes_
        print(classification_report(y_specific_test, specific_preds, 
                                  target_names=specific_names, zero_division=0))
    
    def predict_with_confidence(self, user_features):
        """
        Make predictions with confidence-based specificity control
        
        Args:
            user_features: dict of engineered features for a single user
        
        Returns:
            dict with predictions at different specificity levels
        """
        # Convert to DataFrame
        X_user = pd.DataFrame([user_features])
        
        # Fill missing columns with 0
        for col in self.broad_model.feature_names_in_:
            if col not in X_user.columns:
                X_user[col] = 0
        X_user = X_user[self.broad_model.feature_names_in_]
        
        # Get prediction probabilities
        broad_probs = self.broad_model.predict_proba(X_user)[0]
        field_probs = self.field_model.predict_proba(X_user)[0]
        specific_probs = self.specific_model.predict_proba(X_user)[0]
        
        # Get predictions and confidences
        broad_pred_idx = np.argmax(broad_probs)
        field_pred_idx = np.argmax(field_probs)
        specific_pred_idx = np.argmax(specific_probs)
        
        broad_confidence = broad_probs[broad_pred_idx]
        field_confidence = field_probs[field_pred_idx]
        specific_confidence = specific_probs[specific_pred_idx]
        
        # Determine appropriate specificity level
        if specific_confidence >= self.confidence_thresholds['specific']:
            recommendation_level = 'specific'
            primary_prediction = self.label_encoders['specific'].classes_[specific_pred_idx]
            confidence = specific_confidence
        elif field_confidence >= self.confidence_thresholds['field']:
            recommendation_level = 'field'
            primary_prediction = self.label_encoders['field'].classes_[field_pred_idx]
            confidence = field_confidence
        else:
            recommendation_level = 'broad'
            primary_prediction = self.label_encoders['broad'].classes_[broad_pred_idx]
            confidence = broad_confidence
        
        # Get top alternatives
        top_specific_indices = np.argsort(specific_probs)[-3:][::-1]
        alternatives = []
        for idx in top_specific_indices:
            career = self.label_encoders['specific'].classes_[idx]
            conf = specific_probs[idx]
            alternatives.append({'career': career, 'confidence': conf})
        
        result = {
            'primary_recommendation': {
                'career': primary_prediction,
                'confidence': confidence,
                'level': recommendation_level
            },
            'hierarchical_predictions': {
                'broad': {
                    'category': self.label_encoders['broad'].classes_[broad_pred_idx],
                    'confidence': broad_confidence
                },
                'field': {
                    'category': self.label_encoders['field'].classes_[field_pred_idx],
                    'confidence': field_confidence
                },
                'specific': {
                    'category': self.label_encoders['specific'].classes_[specific_pred_idx],
                    'confidence': specific_confidence
                }
            },
            'top_alternatives': alternatives,
            'recommendation_reasoning': self.get_feature_importance_explanation(X_user)
        }
        
        return result
    
    def get_feature_importance_explanation(self, X_user):
        """Generate explanation based on feature importance"""
        # Get feature importance from specific model
        feature_names = X_user.columns
        feature_values = X_user.iloc[0].values
        
        # Get SHAP values for explanation
        explainer = shap.TreeExplainer(self.specific_model)
        shap_values = explainer.shap_values(X_user)
        
        # Find top contributing features
        if isinstance(shap_values, list) and len(shap_values) > 0:
            # Multi-class case - shap_values is a list of arrays
            pred_class = self.specific_model.predict(X_user)[0]
            class_shap = shap_values[pred_class][0] if len(shap_values) > pred_class else shap_values[0][0]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            # Multi-class case - shap_values is a 3D array
            pred_class = self.specific_model.predict(X_user)[0]
            class_shap = shap_values[pred_class][0]
        else:
            # Binary case or single array
            class_shap = shap_values[0] if hasattr(shap_values, 'shape') else shap_values
        
        # Get top positive contributors
        feature_contributions = list(zip(feature_names, class_shap, feature_values))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = feature_contributions[:5]
        
        explanations = []
        for feature, shap_val, value in top_features:
            if abs(shap_val) > 0.01:  # Only include significant contributions
                if 'hobby_' in feature:
                    explanations.append(f"Your {feature.replace('hobby_', '')} interests (strength: {value:.2f})")
                elif 'grade_' in feature:
                    explanations.append(f"Your {feature.replace('grade_', '')} academic performance (score: {value:.2f})")
                elif 'personality_' in feature:
                    explanations.append(f"Your {feature.replace('personality_', '')} personality trait (score: {value:.2f})")
                elif 'synergy_' in feature:
                    explanations.append(f"The combination of your interests in {feature.replace('synergy_', '').replace('_', ' and ')}")
        
        return explanations
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'broad_model': self.broad_model,
            'field_model': self.field_model,
            'specific_model': self.specific_model,
            'label_encoders': self.label_encoders,
            'feature_engineer': self.feature_engineer,
            'confidence_thresholds': self.confidence_thresholds
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.broad_model = model_data['broad_model']
        self.field_model = model_data['field_model']
        self.specific_model = model_data['specific_model']
        self.label_encoders = model_data['label_encoders']
        self.feature_engineer = model_data['feature_engineer']
        self.confidence_thresholds = model_data['confidence_thresholds']
        print(f"Model loaded from {filepath}")
        
    def predict_user_career(self, academic_grades, user_hobbies, personality_scores):
        """
        Complete pipeline to predict career for a new user
        
        Args:
            academic_grades: dict with subject grades
            user_hobbies: dict with hobby details
            personality_scores: dict with Big Five scores
        
        Returns:
            dict with prediction results
        """
        # Engineer features
        features = self.feature_engineer.engineer_all_features(
            academic_grades, user_hobbies, personality_scores
        )
        
        # Make prediction
        result = self.predict_with_confidence(features)
        
        return result

# Usage example and testing
if __name__ == "__main__":
    # This would be used with real data
    print("Career Prediction Model initialized successfully!")
    print("To use this model, you need to:")
    print("1. Prepare training data with academic grades, hobbies, personality, and career labels")
    print("2. Train the model using train_hierarchical_models()")
    print("3. Make predictions using predict_user_career()")
