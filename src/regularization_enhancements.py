import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier

class RegularizationEnhancer:
    def __init__(self):
        self.regularization_history = {}
        self.optimal_params = {}
        
    def analyze_bias_variance_tradeoff(self, model, X, y, param_name, param_range):
        """Analyze bias-variance tradeoff for a specific parameter"""
        
        train_scores, validation_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=5, scoring='f1_macro', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        validation_mean = np.mean(validation_scores, axis=1)
        validation_std = np.std(validation_scores, axis=1)
        
        # Find optimal parameter value
        optimal_idx = np.argmax(validation_mean)
        optimal_param = param_range[optimal_idx]
        
        results = {
            'param_range': param_range,
            'train_scores': (train_mean, train_std),
            'validation_scores': (validation_mean, validation_std),
            'optimal_param': optimal_param,
            'optimal_score': validation_mean[optimal_idx]
        }
        
        return results
    
    def enhanced_xgboost_regularization(self, X, y):
        """Enhanced XGBoost with advanced regularization"""
        
        # Test different regularization parameters
        reg_alpha_range = np.logspace(-3, 2, 10)  # L1 regularization
        reg_lambda_range = np.logspace(-3, 2, 10)  # L2 regularization
        
        base_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Analyze L1 regularization (reg_alpha)
        print("Analyzing L1 regularization (reg_alpha)...")
        alpha_results = self.analyze_bias_variance_tradeoff(
            base_model, X, y, 'reg_alpha', reg_alpha_range
        )
        
        # Analyze L2 regularization (reg_lambda)  
        print("Analyzing L2 regularization (reg_lambda)...")
        lambda_results = self.analyze_bias_variance_tradeoff(
            base_model, X, y, 'reg_lambda', reg_lambda_range
        )
        
        # Create optimally regularized model
        optimal_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            reg_alpha=alpha_results['optimal_param'],
            reg_lambda=lambda_results['optimal_param'],
            random_state=42,
            eval_metric='mlogloss'
        )
        
        print(f"Optimal reg_alpha: {alpha_results['optimal_param']:.4f}")
        print(f"Optimal reg_lambda: {lambda_results['optimal_param']:.4f}")
        
        return optimal_model, {
            'alpha_analysis': alpha_results,
            'lambda_analysis': lambda_results
        }
    
    def early_stopping_optimization(self, X_train, y_train, X_val, y_val):
        """Implement early stopping to prevent overfitting"""
        
        model = xgb.XGBClassifier(
            n_estimators=2000,  # Set high, will stop early
            max_depth=6,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        print(f"Early stopping at iteration: {model.best_iteration}")
        print(f"Best validation score: {model.best_score:.4f}")
        
        return model
    
    def learning_curve_analysis(self, model, X, y):
        """Analyze learning curves to detect overfitting/underfitting"""
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, validation_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5, 
            scoring='f1_macro', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        validation_mean = np.mean(validation_scores, axis=1)
        validation_std = np.std(validation_scores, axis=1)
        
        # Detect overfitting (large gap between train and validation)
        final_gap = train_mean[-1] - validation_mean[-1]
        overfitting_detected = final_gap > 0.1
        
        # Detect underfitting (both scores are low)
        underfitting_detected = validation_mean[-1] < 0.6
        
        results = {
            'train_sizes': train_sizes,
            'train_scores': (train_mean, train_std),
            'validation_scores': (validation_mean, validation_std),
            'overfitting_detected': overfitting_detected,
            'underfitting_detected': underfitting_detected,
            'final_gap': final_gap
        }
        
        print(f"Learning curve analysis:")
        print(f"- Overfitting detected: {overfitting_detected}")
        print(f"- Underfitting detected: {underfitting_detected}")
        print(f"- Train-validation gap: {final_gap:.4f}")
        
        return results
    
    def adaptive_regularization(self, X, y, complexity_metric='n_features'):
        """Adaptive regularization based on data complexity"""
        
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Calculate complexity metrics
        complexity_metrics = {
            'n_features': n_features,
            'n_samples': n_samples,
            'n_classes': n_classes,
            'feature_to_sample_ratio': n_features / n_samples,
            'samples_per_class': n_samples / n_classes
        }
        
        # Adaptive regularization based on complexity
        if complexity_metrics['feature_to_sample_ratio'] > 0.5:
            # High-dimensional data - stronger regularization
            reg_alpha = 1.0
            reg_lambda = 2.0
            max_depth = 4
            print("High-dimensional data detected - using strong regularization")
        elif complexity_metrics['samples_per_class'] < 50:
            # Few samples per class - moderate regularization
            reg_alpha = 0.5
            reg_lambda = 1.0
            max_depth = 5
            print("Few samples per class detected - using moderate regularization")
        else:
            # Well-balanced data - light regularization
            reg_alpha = 0.1
            reg_lambda = 0.5
            max_depth = 6
            print("Well-balanced data - using light regularization")
        
        adaptive_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=max_depth,
            learning_rate=0.1,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        return adaptive_model, complexity_metrics
    
    def feature_dropout_regularization(self, X, y, dropout_rates=[0.1, 0.2, 0.3]):
        """Implement feature dropout as regularization technique"""
        
        results = {}
        
        for dropout_rate in dropout_rates:
            print(f"Testing feature dropout rate: {dropout_rate}")
            
            # Randomly drop features
            n_features_to_keep = int(X.shape[1] * (1 - dropout_rate))
            random_features = np.random.choice(
                X.shape[1], n_features_to_keep, replace=False
            )
            
            X_dropped = X.iloc[:, random_features]
            
            # Train model with dropped features
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mlogloss'
            )
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_dropped, y, cv=5, scoring='f1_macro')
            
            results[dropout_rate] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'selected_features': X.columns[random_features].tolist()
            }
            
            print(f"Dropout {dropout_rate}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Find optimal dropout rate
        optimal_dropout = max(results.keys(), key=lambda x: results[x]['mean_score'])
        
        print(f"Optimal dropout rate: {optimal_dropout}")
        
        return results, optimal_dropout
    
    def elastic_net_feature_selection(self, X, y, alpha_range=None):
        """Use Elastic Net for feature selection as regularization"""
        
        if alpha_range is None:
            alpha_range = np.logspace(-3, 1, 20)
        
        # For multi-class, we'll use the approach with label binarization
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import ElasticNet
        
        best_alpha = None
        best_score = -np.inf
        selected_features = None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for alpha in alpha_range:
            # Use ElasticNet with OneVsRest for multi-class
            elastic_net = OneVsRestClassifier(
                ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=1000)
            )
            
            try:
                elastic_net.fit(X_scaled, y)
                
                # Get feature importance (average across all binary classifiers)
                if hasattr(elastic_net, 'estimators_'):
                    feature_importance = np.mean([
                        np.abs(est.coef_) for est in elastic_net.estimators_
                    ], axis=0)
                    
                    # Select features with non-zero coefficients
                    selected_mask = feature_importance > 1e-5
                    
                    if np.sum(selected_mask) > 0:
                        X_selected = X.iloc[:, selected_mask]
                        
                        # Evaluate with selected features
                        eval_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
                        from sklearn.model_selection import cross_val_score
                        scores = cross_val_score(eval_model, X_selected, y, cv=3, scoring='f1_macro')
                        
                        if scores.mean() > best_score:
                            best_score = scores.mean()
                            best_alpha = alpha
                            selected_features = X.columns[selected_mask].tolist()
            
            except Exception as e:
                print(f"Error with alpha {alpha}: {e}")
                continue
        
        results = {
            'best_alpha': best_alpha,
            'best_score': best_score,
            'selected_features': selected_features,
            'n_selected_features': len(selected_features) if selected_features else 0
        }
        
        print(f"Elastic Net feature selection:")
        print(f"- Best alpha: {best_alpha}")
        print(f"- Best score: {best_score:.4f}")
        print(f"- Selected features: {len(selected_features) if selected_features else 0}/{X.shape[1]}")
        
        return results

# Integration class for enhanced career predictor
class RegularizedCareerPredictor:
    def __init__(self):
        self.regularization_enhancer = RegularizationEnhancer()
        self.optimal_models = {}
        
    def train_with_enhanced_regularization(self, X, y, model_type='specific'):
        """Train model with enhanced regularization techniques"""
        
        print(f"Training {model_type} model with enhanced regularization...")
        
        # 1. Analyze and optimize regularization parameters
        optimal_model, reg_analysis = self.regularization_enhancer.enhanced_xgboost_regularization(X, y)
        
        # 2. Adaptive regularization based on data complexity
        adaptive_model, complexity_metrics = self.regularization_enhancer.adaptive_regularization(X, y)
        
        # 3. Learning curve analysis
        learning_analysis = self.regularization_enhancer.learning_curve_analysis(optimal_model, X, y)
        
        # 4. Feature selection with Elastic Net
        feature_selection_results = self.regularization_enhancer.elastic_net_feature_selection(X, y)
        
        # Choose the best approach based on analysis
        if learning_analysis['overfitting_detected']:
            print("Overfitting detected - using adaptive model with stronger regularization")
            final_model = adaptive_model
        else:
            print("Using optimally regularized model")
            final_model = optimal_model
        
        # Train final model
        final_model.fit(X, y)
        
        self.optimal_models[model_type] = {
            'model': final_model,
            'regularization_analysis': reg_analysis,
            'complexity_metrics': complexity_metrics,
            'learning_analysis': learning_analysis,
            'feature_selection': feature_selection_results
        }
        
        return final_model
    
    def get_regularization_summary(self, model_type='specific'):
        """Get summary of regularization techniques applied"""
        if model_type in self.optimal_models:
            return self.optimal_models[model_type]
        return None

# Usage example
if __name__ == "__main__":
    print("Enhanced Regularization Techniques:")
    print("1. Bias-variance tradeoff analysis")
    print("2. Early stopping optimization")
    print("3. Learning curve analysis")
    print("4. Adaptive regularization")
    print("5. Feature dropout regularization")
    print("6. Elastic Net feature selection")
