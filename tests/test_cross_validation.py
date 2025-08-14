"""
Unit tests for enhanced_cross_validation.py
Tests the EnhancedCrossValidation class and its methods
"""

import unittest
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add src to path
sys.path.append('src')

from src.enhanced_cross_validation import EnhancedCrossValidation, EnhancedCareerPredictorCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class TestEnhancedCrossValidation(unittest.TestCase):
    """Unit tests for EnhancedCrossValidation class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cv = EnhancedCrossValidation()
        
        # Create smaller synthetic test data for faster testing
        X, y = make_classification(
            n_samples=30, n_features=5, n_informative=3, 
            n_redundant=1, n_classes=2, random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)
        
        # Create a simple model
        self.model = RandomForestClassifier(n_estimators=5, random_state=42)
    
    def test_initialization(self):
        """Test CV class initialization"""
        self.assertIsNotNone(self.cv)
        self.assertEqual(self.cv.cv_results, {})
        self.assertEqual(self.cv.best_models, {})
        self.assertEqual(self.cv.evaluation_history, [])
    
    def test_stratified_cross_validation_structure(self):
        """Test stratified cross-validation structure"""
        # Use smaller cv_folds for faster testing
        results = self.cv.stratified_cross_validation(
            self.model, self.X, self.y, cv_folds=2, scoring='f1_macro'
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('f1_macro', results)
        self.assertIn('f1_micro', results)
        self.assertIn('f1_weighted', results)
        self.assertIn('precision_macro', results)
        self.assertIn('recall_macro', results)
        
        # Check each metric structure
        for metric, values in results.items():
            self.assertIn('test_mean', values)
            self.assertIn('test_std', values)
            self.assertIn('train_mean', values)
            self.assertIn('train_std', values)
            self.assertIn('overfitting', values)
            self.assertIn('test_scores', values)
            self.assertIn('train_scores', values)
            
            # Check data types
            self.assertIsInstance(values['test_mean'], float)
            self.assertIsInstance(values['test_std'], float)
            self.assertIsInstance(values['train_mean'], float)
            self.assertIsInstance(values['train_std'], float)
            self.assertIsInstance(values['overfitting'], float)
            self.assertIsInstance(values['test_scores'], np.ndarray)
            self.assertIsInstance(values['train_scores'], np.ndarray)
            
            # Check value ranges
            self.assertGreaterEqual(values['test_mean'], 0.0)
            self.assertLessEqual(values['test_mean'], 1.0)
            self.assertGreaterEqual(values['train_mean'], 0.0)
            self.assertLessEqual(values['train_mean'], 1.0)
            self.assertGreaterEqual(values['test_std'], 0.0)
            self.assertGreaterEqual(values['train_std'], 0.0)
    
    def test_nested_cross_validation_structure(self):
        """Test nested cross-validation structure"""
        param_grid = {
            'n_estimators': [5],
            'max_depth': [3]
        }
        
        # Use smaller cv folds for faster testing
        results = self.cv.nested_cross_validation(
            self.model, param_grid, self.X, self.y, 
            outer_cv=2, inner_cv=2, scoring='f1_macro'
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('mean_score', results)
        self.assertIn('std_score', results)
        self.assertIn('scores', results)
        self.assertIn('best_params', results)
        self.assertIn('confidence_interval', results)
        
        # Check data types
        self.assertIsInstance(results['mean_score'], float)
        self.assertIsInstance(results['std_score'], float)
        self.assertIsInstance(results['scores'], list)
        self.assertIsInstance(results['best_params'], dict)
        self.assertIsInstance(results['confidence_interval'], tuple)
        
        # Check value ranges
        self.assertGreaterEqual(results['mean_score'], 0.0)
        self.assertLessEqual(results['mean_score'], 1.0)
        self.assertGreaterEqual(results['std_score'], 0.0)
        self.assertGreater(len(results['scores']), 0)
    
    def test_time_series_cross_validation(self):
        """Test time series cross-validation structure"""
        # Create time series data
        X_ts = self.X.copy()
        X_ts['time'] = range(len(X_ts))
        
        results = self.cv.time_series_cross_validation(
            self.model, X_ts, self.y, cv_folds=2
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('mean_score', results)
        self.assertIn('std_score', results)
        self.assertIn('scores', results)
        
        # Check data types
        self.assertIsInstance(results['mean_score'], float)
        self.assertIsInstance(results['std_score'], float)
        self.assertIsInstance(results['scores'], list)
        
        # Check value ranges
        self.assertGreaterEqual(results['mean_score'], 0.0)
        self.assertLessEqual(results['mean_score'], 1.0)
        self.assertGreaterEqual(results['std_score'], 0.0)
    
    def test_leave_one_out_cross_validation(self):
        """Test leave-one-out cross-validation structure"""
        # Use smaller dataset for LOO
        X_small = self.X.iloc[:10]
        y_small = self.y.iloc[:10]
        
        results = self.cv.leave_one_out_cross_validation(
            self.model, X_small, y_small
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('mean_accuracy', results)
        self.assertIn('std_accuracy', results)
        self.assertIn('scores', results)
        
        # Check data types
        self.assertIsInstance(results['mean_accuracy'], float)
        self.assertIsInstance(results['std_accuracy'], float)
        self.assertIsInstance(results['scores'], list)
        
        # Check value ranges
        self.assertGreaterEqual(results['mean_accuracy'], 0.0)
        self.assertLessEqual(results['mean_accuracy'], 1.0)
        self.assertGreaterEqual(results['std_accuracy'], 0.0)
    
    def test_group_cross_validation(self):
        """Test group cross-validation structure"""
        # Create groups
        groups = np.array([i % 3 for i in range(len(self.X))])
        
        results = self.cv.group_cross_validation(
            self.model, self.X, self.y, groups, cv_folds=2
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('mean_score', results)
        self.assertIn('std_score', results)
        self.assertIn('scores', results)
        self.assertIn('n_groups', results)
        
        # Check data types
        self.assertIsInstance(results['mean_score'], float)
        self.assertIsInstance(results['std_score'], float)
        self.assertIsInstance(results['scores'], list)
        self.assertIsInstance(results['n_groups'], int)
        
        # Check value ranges
        self.assertGreaterEqual(results['mean_score'], 0.0)
        self.assertLessEqual(results['mean_score'], 1.0)
        self.assertGreaterEqual(results['std_score'], 0.0)
        self.assertGreater(results['n_groups'], 0)
    
    def test_bootstrap_cross_validation(self):
        """Test bootstrap cross-validation structure"""
        # Use smaller number of iterations for faster testing
        results = self.cv.bootstrap_cross_validation(
            self.model, self.X, self.y, n_iterations=5
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('mean_score', results)
        self.assertIn('std_score', results)
        self.assertIn('all_scores', results)
        self.assertIn('confidence_interval', results)
        self.assertIn('min_score', results)
        self.assertIn('max_score', results)
        
        # Check data types and ranges
        self.assertIsInstance(results['all_scores'], list)
        self.assertIsInstance(results['mean_score'], float)
        self.assertIsInstance(results['std_score'], float)
        self.assertIsInstance(results['min_score'], float)
        self.assertIsInstance(results['max_score'], float)
        
        self.assertGreaterEqual(results['mean_score'], 0.0)
        self.assertLessEqual(results['mean_score'], 1.0)
        self.assertGreaterEqual(results['std_score'], 0.0)
        self.assertLessEqual(results['min_score'], results['max_score'])
    
    def test_repeated_cross_validation(self):
        """Test repeated cross-validation structure"""
        # Use smaller parameters for faster testing
        results = self.cv.repeated_cross_validation(
            self.model, self.X, self.y, n_repeats=2, cv_folds=2
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('mean_score', results)
        self.assertIn('std_score', results)
        self.assertIn('all_scores', results)
        self.assertIn('confidence_interval', results)
        
        # Check data types
        self.assertIsInstance(results['mean_score'], float)
        self.assertIsInstance(results['std_score'], float)
        self.assertIsInstance(results['all_scores'], list)
        self.assertIsInstance(results['confidence_interval'], tuple)
        
        # Check value ranges
        self.assertGreaterEqual(results['mean_score'], 0.0)
        self.assertLessEqual(results['mean_score'], 1.0)
        self.assertGreaterEqual(results['std_score'], 0.0)
    
    @patch('builtins.print')  # Mock print to reduce output
    def test_comprehensive_cv_comparison(self, mock_print):
        """Test comprehensive CV comparison structure"""
        param_grid = {'n_estimators': [5]}
        
        results = self.cv.comprehensive_cv_comparison(
            self.model, self.X, self.y, param_grid
        )
        
        # Check results structure
        self.assertIn('stratified_cv', results)
        self.assertIn('nested_cv', results)
        self.assertIn('bootstrap_cv', results)
        self.assertIn('repeated_cv', results)
        self.assertIn('recommendations', results)
        
        # Check that each CV method has results
        for method in ['stratified_cv', 'nested_cv', 'bootstrap_cv', 'repeated_cv']:
            self.assertIsInstance(results[method], dict)
            self.assertGreater(len(results[method]), 0)
        
        # Check recommendations
        self.assertIsInstance(results['recommendations'], list)
        self.assertGreater(len(results['recommendations']), 0)
    
    def test_get_cv_recommendations(self):
        """Test CV recommendations generation"""
        # First run some CV to populate results
        self.cv.stratified_cross_validation(
            self.model, self.X, self.y, cv_folds=2
        )
        
        recommendations = self.cv.get_cv_recommendations()
        
        # Check recommendations structure
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check each recommendation is a string
        for rec in recommendations:
            self.assertIsInstance(rec, str)
            self.assertGreater(len(rec), 0)


class TestEnhancedCareerPredictorCV(unittest.TestCase):
    """Unit tests for EnhancedCareerPredictorCV class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_predictor = Mock()
        self.cv_predictor = EnhancedCareerPredictorCV(self.base_predictor)
        
        # Create smaller synthetic test data
        X, y = make_classification(
            n_samples=20, n_features=5, n_informative=3, 
            n_redundant=1, n_classes=2, random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)
    
    def test_initialization(self):
        """Test CV predictor initialization"""
        self.assertIsNotNone(self.cv_predictor)
        self.assertEqual(self.cv_predictor.base_predictor, self.base_predictor)
    
    @patch('builtins.print')  # Mock print to reduce output
    def test_comprehensive_evaluation(self, mock_print):
        """Test comprehensive evaluation"""
        param_grid = {'n_estimators': [5]}
        
        results = self.cv_predictor.comprehensive_evaluation(
            self.X, self.y, param_grid
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('cv_results', results)
        self.assertIn('best_params', results)
        self.assertIn('recommendations', results)
        
        # Check that CV results exist
        self.assertIsInstance(results['cv_results'], dict)
        self.assertGreater(len(results['cv_results']), 0)


if __name__ == '__main__':
    unittest.main() 