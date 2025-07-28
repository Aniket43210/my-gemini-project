import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import numpy as np

class AdvancedHyperparameterTuner:
    def __init__(self):
        self.best_params_history = {}
        
    def enhanced_optimize_hyperparameters(self, X, y, model_type='specific', n_trials=100):
        """Enhanced hyperparameter optimization with expanded search space"""
        
        def objective(trial):
            # Expanded parameter space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10),
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Model-specific parameter adjustments
            if model_type == 'broad':
                # For broad categories, use simpler models
                params['max_depth'] = min(params['max_depth'], 8)
                params['n_estimators'] = min(params['n_estimators'], 1000)
            elif model_type == 'specific':
                # For specific careers, allow more complex models
                params['max_depth'] = trial.suggest_int('max_depth', 4, 20)
                params['n_estimators'] = trial.suggest_int('n_estimators', 200, 3000)
            
            model = xgb.XGBClassifier(**params)
            
            # Use stratified k-fold for better evaluation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Use F1-macro score for multi-class problems
            f1_scorer = make_scorer(f1_score, average='macro')
            scores = cross_val_score(model, X, y, cv=cv, scoring=f1_scorer, n_jobs=-1)
            
            return scores.mean()
        
        # Create study with pruning for efficiency
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store best parameters
        self.best_params_history[model_type] = study.best_params
        
        print(f"Best {model_type} model score: {study.best_value:.4f}")
        print(f"Best {model_type} parameters: {study.best_params}")
        
        return study.best_params
    
    def multi_objective_optimization(self, X, y, model_type='specific', n_trials=50):
        """Multi-objective optimization balancing accuracy and model complexity"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # Objective 1: Accuracy (F1-score)
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
            accuracy_objective = f1_scores.mean()
            
            # Objective 2: Model complexity (inverse of tree depth and number of estimators)
            complexity_penalty = (params['max_depth'] / 12.0) + (params['n_estimators'] / 1500.0)
            complexity_objective = 1 - (complexity_penalty / 2.0)  # Lower complexity is better
            
            return accuracy_objective, complexity_objective
        
        study = optuna.create_study(
            directions=['maximize', 'maximize'],  # Both objectives to maximize
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best solution from Pareto front
        pareto_solutions = study.best_trials
        if pareto_solutions:
            best_trial = max(pareto_solutions, key=lambda t: t.values[0])  # Prioritize accuracy
            return best_trial.params
        
        return {}

# Usage example for integration into existing code
class EnhancedCareerPredictor:
    def __init__(self):
        self.tuner = AdvancedHyperparameterTuner()
        # ... other initialization code
    
    def optimize_hyperparameters_enhanced(self, X, y, model_type='specific'):
        """Enhanced version to replace the existing method"""
        return self.tuner.enhanced_optimize_hyperparameters(X, y, model_type, n_trials=100)
    
    def get_tuning_history(self):
        """Get the history of best parameters for analysis"""
        return self.tuner.best_params_history
