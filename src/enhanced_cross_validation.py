import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, TimeSeriesSplit, LeaveOneOut,
    cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class EnhancedCrossValidation:
    def __init__(self):
        self.cv_results = {}
        self.best_models = {}
        self.evaluation_history = []
        
    def stratified_cross_validation(self, model, X, y, cv_folds=5, scoring='f1_macro'):
        """Enhanced stratified cross-validation with detailed metrics"""
        
        # Ensure stratification is possible
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Define multiple scoring metrics
        scoring_metrics = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'f1_micro': 'f1_micro',
            'f1_weighted': 'f1_weighted',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro'
        }
        
        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            model, X, y_encoded, cv=skf, scoring=scoring_metrics,
            return_train_score=True, n_jobs=-1
        )
        
        # Calculate statistics
        results = {}
        for metric in scoring_metrics.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'overfitting': np.mean(train_scores) - np.mean(test_scores),
                'test_scores': test_scores,
                'train_scores': train_scores
            }
        
        # Store results
        self.cv_results['stratified'] = results
        
        # Print summary
        print("Stratified Cross-Validation Results:")
        print("="*50)
        for metric, values in results.items():
            print(f"{metric.upper()}:")
            print(f"  Test:  {values['test_mean']:.4f} (+/- {values['test_std']*2:.4f})")
            print(f"  Train: {values['train_mean']:.4f} (+/- {values['train_std']*2:.4f})")
            print(f"  Overfitting Gap: {values['overfitting']:.4f}")
            print()
        
        return results
    
    def nested_cross_validation(self, model, param_grid, X, y, 
                               outer_cv=5, inner_cv=3, scoring='f1_macro'):
        """Implement nested cross-validation for unbiased model evaluation"""
        
        print("Performing Nested Cross-Validation...")
        print(f"Outer CV: {outer_cv} folds, Inner CV: {inner_cv} folds")
        
        # Outer cross-validation
        outer_cv_obj = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
        
        # Store results for each outer fold
        outer_scores = []
        best_params_list = []
        
        fold_num = 1
        for train_idx, test_idx in outer_cv_obj.split(X, y):
            print(f"Processing outer fold {fold_num}/{outer_cv}...")
            
            # Split data for this outer fold
            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test_outer = np.array(y)[train_idx], np.array(y)[test_idx]
            
            # Inner cross-validation for hyperparameter tuning
            inner_cv_obj = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
            
            # Perform grid search on training data
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv_obj, scoring=scoring,
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Get best model and evaluate on test set
            best_model = grid_search.best_estimator_
            best_params_list.append(grid_search.best_params_)
            
            # Evaluate on outer test set
            test_score = best_model.score(X_test_outer, y_test_outer)
            outer_scores.append(test_score)
            
            print(f"  Fold {fold_num} score: {test_score:.4f}")
            print(f"  Best params: {grid_search.best_params_}")
            
            fold_num += 1
        
        # Calculate final statistics
        nested_cv_results = {
            'outer_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params_list': best_params_list,
            'confidence_interval': (
                np.mean(outer_scores) - 1.96 * np.std(outer_scores) / np.sqrt(outer_cv),
                np.mean(outer_scores) + 1.96 * np.std(outer_scores) / np.sqrt(outer_cv)
            )
        }
        
        print("\nNested Cross-Validation Final Results:")
        print(f"Mean Score: {nested_cv_results['mean_score']:.4f}")
        print(f"Std Score: {nested_cv_results['std_score']:.4f}")
        print(f"95% Confidence Interval: {nested_cv_results['confidence_interval']}")
        
        self.cv_results['nested'] = nested_cv_results
        return nested_cv_results
    
    def time_series_cross_validation(self, model, X, y, n_splits=5, test_size=None):
        """Time series cross-validation for temporal data"""
        
        if test_size is None:
            test_size = len(X) // (n_splits + 1)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        scores = cross_val_score(model, X, y, cv=tscv, scoring='f1_macro')
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_splits': n_splits,
            'test_size': test_size
        }
        
        print("Time Series Cross-Validation Results:")
        print(f"Scores: {scores}")
        print(f"Mean: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        
        self.cv_results['time_series'] = results
        return results
    
    def leave_one_out_cross_validation(self, model, X, y):
        """Leave-One-Out cross-validation for small datasets"""
        
        if len(X) > 1000:
            print("Warning: LOO-CV with large dataset. Consider using other CV methods.")
            return None
        
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy', n_jobs=-1)
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_samples': len(X)
        }
        
        print("Leave-One-Out Cross-Validation Results:")
        print(f"Mean Accuracy: {results['mean_score']:.4f}")
        print(f"Std: {results['std_score']:.4f}")
        
        self.cv_results['loo'] = results
        return results
    
    def group_cross_validation(self, model, X, y, groups, cv_folds=5):
        """Group cross-validation to prevent data leakage"""
        
        group_cv = GroupKFold(n_splits=cv_folds)
        scores = cross_val_score(model, X, y, groups=groups, cv=group_cv, scoring='f1_macro')
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_groups': len(np.unique(groups))
        }
        
        print("Group Cross-Validation Results:")
        print(f"Scores: {scores}")
        print(f"Mean: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        print(f"Number of groups: {results['n_groups']}")
        
        self.cv_results['group'] = results
        return results
    
    def bootstrap_cross_validation(self, model, X, y, n_bootstraps=100):
        """Bootstrap cross-validation for robust evaluation"""
        
        n_samples = len(X)
        bootstrap_scores = []
        
        print(f"Performing Bootstrap Cross-Validation with {n_bootstraps} iterations...")
        
        for i in range(n_bootstraps):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            out_of_bag_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            
            if len(out_of_bag_indices) == 0:
                continue
            
            # Train on bootstrap sample
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = np.array(y)[bootstrap_indices]
            
            # Test on out-of-bag samples
            X_oob = X.iloc[out_of_bag_indices]
            y_oob = np.array(y)[out_of_bag_indices]
            
            # Fit and evaluate
            model.fit(X_bootstrap, y_bootstrap)
            score = model.score(X_oob, y_oob)
            bootstrap_scores.append(score)
        
        results = {
            'scores': bootstrap_scores,
            'mean_score': np.mean(bootstrap_scores),
            'std_score': np.std(bootstrap_scores),
            'confidence_interval': np.percentile(bootstrap_scores, [2.5, 97.5]),
            'n_bootstraps': len(bootstrap_scores)
        }
        
        print("Bootstrap Cross-Validation Results:")
        print(f"Mean Score: {results['mean_score']:.4f}")
        print(f"Std Score: {results['std_score']:.4f}")
        print(f"95% CI: {results['confidence_interval']}")
        
        self.cv_results['bootstrap'] = results
        return results
    
    def repeated_cross_validation(self, model, X, y, cv_folds=5, n_repeats=10):
        """Repeated cross-validation for more stable estimates"""
        
        all_scores = []
        
        print(f"Performing Repeated Cross-Validation ({n_repeats} repeats, {cv_folds} folds each)...")
        
        for repeat in range(n_repeats):
            # Create new random state for each repeat
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=repeat)
            scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
            all_scores.extend(scores)
            
            if repeat % 2 == 0:
                print(f"Completed repeat {repeat + 1}/{n_repeats}")
        
        results = {
            'all_scores': all_scores,
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'n_total_folds': len(all_scores),
            'confidence_interval': (
                np.mean(all_scores) - 1.96 * np.std(all_scores) / np.sqrt(len(all_scores)),
                np.mean(all_scores) + 1.96 * np.std(all_scores) / np.sqrt(len(all_scores))
            )
        }
        
        print("Repeated Cross-Validation Results:")
        print(f"Mean Score: {results['mean_score']:.4f}")
        print(f"Std Score: {results['std_score']:.4f}")
        print(f"95% CI: {results['confidence_interval']}")
        
        self.cv_results['repeated'] = results
        return results
    
    def comprehensive_cv_comparison(self, model, X, y, param_grid=None):
        """Compare multiple cross-validation strategies"""
        
        print("Comprehensive Cross-Validation Comparison")
        print("="*60)
        
        comparison_results = {}
        
        # 1. Standard Stratified CV
        print("\n1. Stratified Cross-Validation:")
        comparison_results['stratified'] = self.stratified_cross_validation(model, X, y)
        
        # 2. Repeated CV
        print("\n2. Repeated Cross-Validation:")
        comparison_results['repeated'] = self.repeated_cross_validation(model, X, y)
        
        # 3. Bootstrap CV
        print("\n3. Bootstrap Cross-Validation:")
        comparison_results['bootstrap'] = self.bootstrap_cross_validation(model, X, y)
        
        # 4. Nested CV (if param_grid provided)
        if param_grid:
            print("\n4. Nested Cross-Validation:")
            comparison_results['nested'] = self.nested_cross_validation(
                model, param_grid, X, y
            )
        
        # Create summary comparison
        summary = {}
        for cv_type, results in comparison_results.items():
            if cv_type == 'stratified':
                # Use f1_macro for stratified CV
                mean_score = results['f1_macro']['test_mean']
                std_score = results['f1_macro']['test_std']
            else:
                mean_score = results['mean_score']
                std_score = results['std_score']
            
            summary[cv_type] = {
                'mean': mean_score,
                'std': std_score,
                'lower_ci': mean_score - 1.96 * std_score,
                'upper_ci': mean_score + 1.96 * std_score
            }
        
        # Print comparison summary
        print("\n" + "="*60)
        print("CROSS-VALIDATION COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'95% CI':<20}")
        print("-" * 60)
        
        for method, stats in summary.items():
            ci_str = f"[{stats['lower_ci']:.3f}, {stats['upper_ci']:.3f}]"
            print(f"{method.capitalize():<15} {stats['mean']:<8.4f} {stats['std']:<8.4f} {ci_str:<20}")
        
        return comparison_results, summary
    
    def get_cv_recommendations(self):
        """Get recommendations based on CV results"""
        
        recommendations = []
        
        if 'stratified' in self.cv_results:
            stratified = self.cv_results['stratified']
            f1_results = stratified.get('f1_macro', {})
            
            overfitting = f1_results.get('overfitting', 0)
            if overfitting > 0.1:
                recommendations.append("High overfitting detected. Consider:")
                recommendations.append("- Increase regularization")
                recommendations.append("- Reduce model complexity")
                recommendations.append("- Collect more training data")
            
            test_std = f1_results.get('test_std', 0)
            if test_std > 0.1:
                recommendations.append("High variance in CV scores. Consider:")
                recommendations.append("- Use repeated cross-validation")
                recommendations.append("- Increase number of CV folds")
                recommendations.append("- Ensure data is well-shuffled")
        
        if 'nested' in self.cv_results:
            nested = self.cv_results['nested']
            if nested['std_score'] > 0.05:
                recommendations.append("Unstable hyperparameter selection. Consider:")
                recommendations.append("- Increase inner CV folds")
                recommendations.append("- Use randomized search instead of grid search")
        
        return recommendations

# Integration class
class EnhancedCareerPredictorCV:
    def __init__(self, base_predictor):
        self.base_predictor = base_predictor
        self.cv_evaluator = EnhancedCrossValidation()
        
    def comprehensive_evaluation(self, X, y, param_grid=None):
        """Perform comprehensive cross-validation evaluation"""
        
        print("Starting Comprehensive Model Evaluation...")
        
        # Use the specific model from hierarchical predictor
        if hasattr(self.base_predictor, 'specific_model') and self.base_predictor.specific_model:
            model = self.base_predictor.specific_model
        else:
            print("Warning: No trained model found. Training basic model...")
            import xgboost as xgb
            model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        
        # Perform comprehensive CV comparison
        results, summary = self.cv_evaluator.comprehensive_cv_comparison(
            model, X, y, param_grid
        )
        
        # Get recommendations
        recommendations = self.cv_evaluator.get_cv_recommendations()
        
        if recommendations:
            print("\n" + "="*60)
            print("RECOMMENDATIONS")
            print("="*60)
            for rec in recommendations:
                print(f"• {rec}")
        
        return results, summary, recommendations

# Usage example
if __name__ == "__main__":
    print("Enhanced Cross-Validation System")
    print("Features:")
    print("- Stratified cross-validation with multiple metrics")
    print("- Nested cross-validation for unbiased evaluation")
    print("- Bootstrap cross-validation")
    print("- Repeated cross-validation")
    print("- Group cross-validation")
    print("- Time series cross-validation")
    print("- Comprehensive comparison and recommendations")
