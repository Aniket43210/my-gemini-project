import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

class EnsembleCareerPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ensemble_models = {}
        self.voting_ensemble = None
        self.stacking_ensemble = None
        self.model_weights = {}
        
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        base_models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,  # Enable probability estimates
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            'naive_bayes': GaussianNB()
        }
        return base_models
    
    def create_voting_ensemble(self, X, y, model_type='soft'):
        """Create voting ensemble with optimized weights"""
        base_models = self.create_base_models()
        
        # Evaluate individual models to determine weights
        model_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("Evaluating base models for ensemble weights...")
        for name, model in base_models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
                model_scores[name] = scores.mean()
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                model_scores[name] = 0.0
        
        # Convert scores to weights (higher score = higher weight)
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = [model_scores[name] / total_score for name in base_models.keys()]
        else:
            weights = [1/len(base_models)] * len(base_models)  # Equal weights if all failed
        
        # Create voting ensemble
        estimators = [(name, model) for name, model in base_models.items()]
        
        if model_type == 'soft':
            voting_ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights
            )
        else:
            voting_ensemble = VotingClassifier(
                estimators=estimators,
                voting='hard'
            )
        
        self.model_weights = dict(zip(base_models.keys(), weights))
        return voting_ensemble
    
    def create_stacking_ensemble(self, X, y):
        """Create stacking ensemble with meta-learner"""
        base_models = self.create_base_models()
        
        # Use logistic regression as meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # 5-fold cross-validation for base model predictions
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_ensemble
    
    def create_custom_weighted_ensemble(self, models, weights):
        """Create custom weighted ensemble"""
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = np.array(weights)
                self.weights = self.weights / self.weights.sum()  # Normalize weights
                
            def fit(self, X, y):
                for model in self.models:
                    model.fit(X, y)
                return self
                
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models])
                # Weighted majority vote
                weighted_preds = []
                for i in range(X.shape[0]):
                    votes = {}
                    for j, pred in enumerate(predictions[:, i]):
                        votes[pred] = votes.get(pred, 0) + self.weights[j]
                    weighted_preds.append(max(votes, key=votes.get))
                return np.array(weighted_preds)
                
            def predict_proba(self, X):
                probabilities = np.array([model.predict_proba(X) for model in self.models])
                weighted_probs = np.average(probabilities, axis=0, weights=self.weights)
                return weighted_probs
        
        return WeightedEnsemble(models, weights)
    
    def train_ensemble_models(self, X, y, ensemble_types=['voting', 'stacking']):
        """Train multiple ensemble models"""
        # Scale features for models that need it
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        results = {}
        
        if 'voting' in ensemble_types:
            print("\nTraining Voting Ensemble...")
            self.voting_ensemble = self.create_voting_ensemble(X_scaled, y)
            self.voting_ensemble.fit(X_scaled, y)
            
            # Evaluate voting ensemble
            cv_scores = cross_val_score(self.voting_ensemble, X_scaled, y, 
                                      cv=5, scoring='f1_macro', n_jobs=-1)
            results['voting'] = {
                'model': self.voting_ensemble,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"Voting Ensemble CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if 'stacking' in ensemble_types:
            print("\nTraining Stacking Ensemble...")
            self.stacking_ensemble = self.create_stacking_ensemble(X_scaled, y)
            self.stacking_ensemble.fit(X_scaled, y)
            
            # Evaluate stacking ensemble
            cv_scores = cross_val_score(self.stacking_ensemble, X_scaled, y, 
                                      cv=5, scoring='f1_macro', n_jobs=-1)
            results['stacking'] = {
                'model': self.stacking_ensemble,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"Stacking Ensemble CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best ensemble
        if results:
            best_ensemble_name = max(results, key=lambda x: results[x]['cv_score'])
            self.best_ensemble = results[best_ensemble_name]['model']
            print(f"\nBest Ensemble: {best_ensemble_name} with score {results[best_ensemble_name]['cv_score']:.4f}")
        
        return results
    
    def predict_with_ensemble(self, X, ensemble_type='best'):
        """Make predictions using ensemble model"""
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        if ensemble_type == 'voting' and self.voting_ensemble:
            return self.voting_ensemble.predict(X_scaled)
        elif ensemble_type == 'stacking' and self.stacking_ensemble:
            return self.stacking_ensemble.predict(X_scaled)
        elif ensemble_type == 'best' and hasattr(self, 'best_ensemble'):
            return self.best_ensemble.predict(X_scaled)
        else:
            raise ValueError(f"Ensemble type {ensemble_type} not available or not trained")
    
    def predict_proba_with_ensemble(self, X, ensemble_type='best'):
        """Get prediction probabilities from ensemble"""
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        if ensemble_type == 'voting' and self.voting_ensemble:
            return self.voting_ensemble.predict_proba(X_scaled)
        elif ensemble_type == 'stacking' and self.stacking_ensemble:
            return self.stacking_ensemble.predict_proba(X_scaled)
        elif ensemble_type == 'best' and hasattr(self, 'best_ensemble'):
            return self.best_ensemble.predict_proba(X_scaled)
        else:
            raise ValueError(f"Ensemble type {ensemble_type} not available or not trained")
    
    def get_model_importance(self):
        """Get importance of different models in the ensemble"""
        if hasattr(self, 'model_weights'):
            return self.model_weights
        return {}
    
    def save_ensemble(self, filepath):
        """Save the trained ensemble models"""
        ensemble_data = {
            'voting_ensemble': self.voting_ensemble,
            'stacking_ensemble': self.stacking_ensemble,
            'scaler': self.scaler,
            'model_weights': self.model_weights
        }
        
        if hasattr(self, 'best_ensemble'):
            ensemble_data['best_ensemble'] = self.best_ensemble
        
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble models saved to {filepath}")
    
    def load_ensemble(self, filepath):
        """Load trained ensemble models"""
        ensemble_data = joblib.load(filepath)
        
        self.voting_ensemble = ensemble_data.get('voting_ensemble')
        self.stacking_ensemble = ensemble_data.get('stacking_ensemble')
        self.scaler = ensemble_data.get('scaler')
        self.model_weights = ensemble_data.get('model_weights', {})
        
        if 'best_ensemble' in ensemble_data:
            self.best_ensemble = ensemble_data['best_ensemble']
        
        print(f"Ensemble models loaded from {filepath}")

# Integration with existing career predictor
class HierarchicalEnsembleCareerPredictor:
    """Enhanced version of the hierarchical predictor using ensembles"""
    
    def __init__(self):
        self.broad_ensemble = EnsembleCareerPredictor()
        self.field_ensemble = EnsembleCareerPredictor()
        self.specific_ensemble = EnsembleCareerPredictor()
        self.label_encoders = {}
        
    def train_hierarchical_ensembles(self, X, y_specific):
        """Train ensemble models at different hierarchy levels"""
        # Create hierarchical labels (this would use the existing method)
        # For demonstration, assuming we have the labels
        
        print("Training hierarchical ensemble models...")
        
        # Train ensembles for each level
        broad_results = self.broad_ensemble.train_ensemble_models(X, y_specific)
        field_results = self.field_ensemble.train_ensemble_models(X, y_specific)
        specific_results = self.specific_ensemble.train_ensemble_models(X, y_specific)
        
        return {
            'broad': broad_results,
            'field': field_results,
            'specific': specific_results
        }

# Usage example
if __name__ == "__main__":
    # Example usage
    print("Ensemble Career Predictor initialized!")
    print("Features:")
    print("- Voting ensemble with weighted models")
    print("- Stacking ensemble with meta-learner")
    print("- Custom weighted ensemble")
    print("- Hierarchical ensemble support")
