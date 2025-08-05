"""
Unit tests for ensemble_models.py
Tests the ensemble model functionality
"""

import unittest
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add src to path
sys.path.append('src')

from src.ensemble_models import (
    create_ensemble_model,
    create_weighted_ensemble,
    create_stacking_ensemble,
    create_blending_ensemble,
    evaluate_ensemble_performance,
    EnsembleModelManager
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification


class TestEnsembleModels(unittest.TestCase):
    """Unit tests for ensemble model functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic test data
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, 
            n_redundant=2, n_classes=3, random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)
        
        # Create base models
        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42)
        }
    
    def test_create_ensemble_model(self):
        """Test basic ensemble model creation"""
        ensemble = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        
        # Check ensemble structure
        self.assertIsNotNone(ensemble)
        self.assertIsInstance(ensemble, object)
        
        # Check that ensemble has required attributes
        self.assertTrue(hasattr(ensemble, 'fit'))
        self.assertTrue(hasattr(ensemble, 'predict'))
        self.assertTrue(hasattr(ensemble, 'predict_proba'))
    
    def test_create_weighted_ensemble(self):
        """Test weighted ensemble creation"""
        weights = {'rf': 0.5, 'lr': 0.3, 'svm': 0.2}
        
        ensemble = create_weighted_ensemble(
            self.base_models, weights
        )
        
        # Check ensemble structure
        self.assertIsNotNone(ensemble)
        self.assertIsInstance(ensemble, object)
        
        # Check that ensemble has required attributes
        self.assertTrue(hasattr(ensemble, 'fit'))
        self.assertTrue(hasattr(ensemble, 'predict'))
        self.assertTrue(hasattr(ensemble, 'predict_proba'))
    
    def test_create_stacking_ensemble(self):
        """Test stacking ensemble creation"""
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        ensemble = create_stacking_ensemble(
            self.base_models, meta_model, cv_folds=3
        )
        
        # Check ensemble structure
        self.assertIsNotNone(ensemble)
        self.assertIsInstance(ensemble, object)
        
        # Check that ensemble has required attributes
        self.assertTrue(hasattr(ensemble, 'fit'))
        self.assertTrue(hasattr(ensemble, 'predict'))
        self.assertTrue(hasattr(ensemble, 'predict_proba'))
    
    def test_create_blending_ensemble(self):
        """Test blending ensemble creation"""
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        ensemble = create_blending_ensemble(
            self.base_models, meta_model, validation_split=0.3
        )
        
        # Check ensemble structure
        self.assertIsNotNone(ensemble)
        self.assertIsInstance(ensemble, object)
        
        # Check that ensemble has required attributes
        self.assertTrue(hasattr(ensemble, 'fit'))
        self.assertTrue(hasattr(ensemble, 'predict'))
        self.assertTrue(hasattr(ensemble, 'predict_proba'))
    
    def test_evaluate_ensemble_performance(self):
        """Test ensemble performance evaluation"""
        # Create a simple ensemble
        ensemble = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        
        # Fit the ensemble
        ensemble.fit(self.X, self.y)
        
        # Evaluate performance
        results = evaluate_ensemble_performance(
            ensemble, self.X, self.y, cv_folds=3
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)
        self.assertIn('cv_scores', results)
        
        # Check data types and ranges
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertIsInstance(results[metric], float)
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)
        
        self.assertIsInstance(results['cv_scores'], list)
        self.assertGreater(len(results['cv_scores']), 0)
        
        # Check CV scores are within valid range
        for score in results['cv_scores']:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestEnsembleModelManager(unittest.TestCase):
    """Unit tests for EnsembleModelManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = EnsembleModelManager()
        
        # Create synthetic test data
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, 
            n_redundant=2, n_classes=3, random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)
        
        # Create base models
        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42)
        }
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.ensembles, {})
        self.assertEqual(self.manager.performance_history, [])
    
    def test_add_ensemble(self):
        """Test adding ensemble to manager"""
        ensemble = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        
        self.manager.add_ensemble('test_ensemble', ensemble)
        
        # Check ensemble was added
        self.assertIn('test_ensemble', self.manager.ensembles)
        self.assertEqual(self.manager.ensembles['test_ensemble'], ensemble)
    
    def test_remove_ensemble(self):
        """Test removing ensemble from manager"""
        ensemble = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        
        self.manager.add_ensemble('test_ensemble', ensemble)
        self.manager.remove_ensemble('test_ensemble')
        
        # Check ensemble was removed
        self.assertNotIn('test_ensemble', self.manager.ensembles)
    
    def test_get_ensemble(self):
        """Test getting ensemble from manager"""
        ensemble = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        
        self.manager.add_ensemble('test_ensemble', ensemble)
        retrieved_ensemble = self.manager.get_ensemble('test_ensemble')
        
        # Check retrieved ensemble
        self.assertEqual(retrieved_ensemble, ensemble)
    
    def test_get_ensemble_nonexistent(self):
        """Test getting nonexistent ensemble"""
        with self.assertRaises(KeyError):
            self.manager.get_ensemble('nonexistent_ensemble')
    
    def test_list_ensembles(self):
        """Test listing all ensembles"""
        ensemble1 = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        ensemble2 = create_ensemble_model(
            self.base_models, voting_method='hard'
        )
        
        self.manager.add_ensemble('ensemble1', ensemble1)
        self.manager.add_ensemble('ensemble2', ensemble2)
        
        ensembles = self.manager.list_ensembles()
        
        # Check list contains all ensembles
        self.assertIn('ensemble1', ensembles)
        self.assertIn('ensemble2', ensembles)
        self.assertEqual(len(ensembles), 2)
    
    def test_evaluate_all_ensembles(self):
        """Test evaluating all ensembles"""
        ensemble1 = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        ensemble2 = create_ensemble_model(
            self.base_models, voting_method='hard'
        )
        
        self.manager.add_ensemble('ensemble1', ensemble1)
        self.manager.add_ensemble('ensemble2', ensemble2)
        
        results = self.manager.evaluate_all_ensembles(
            self.X, self.y, cv_folds=3
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('ensemble1', results)
        self.assertIn('ensemble2', results)
        
        # Check each ensemble has evaluation results
        for ensemble_name, ensemble_results in results.items():
            self.assertIsInstance(ensemble_results, dict)
            self.assertIn('accuracy', ensemble_results)
            self.assertIn('precision', ensemble_results)
            self.assertIn('recall', ensemble_results)
            self.assertIn('f1_score', ensemble_results)
            
            # Check data types and ranges
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                self.assertIsInstance(ensemble_results[metric], float)
                self.assertGreaterEqual(ensemble_results[metric], 0.0)
                self.assertLessEqual(ensemble_results[metric], 1.0)
    
    def test_get_best_ensemble(self):
        """Test getting best performing ensemble"""
        ensemble1 = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        ensemble2 = create_ensemble_model(
            self.base_models, voting_method='hard'
        )
        
        self.manager.add_ensemble('ensemble1', ensemble1)
        self.manager.add_ensemble('ensemble2', ensemble2)
        
        # Evaluate all ensembles first
        self.manager.evaluate_all_ensembles(self.X, self.y, cv_folds=3)
        
        best_ensemble_name = self.manager.get_best_ensemble('accuracy')
        
        # Check best ensemble name is valid
        self.assertIn(best_ensemble_name, ['ensemble1', 'ensemble2'])
    
    def test_get_performance_summary(self):
        """Test getting performance summary"""
        ensemble1 = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        ensemble2 = create_ensemble_model(
            self.base_models, voting_method='hard'
        )
        
        self.manager.add_ensemble('ensemble1', ensemble1)
        self.manager.add_ensemble('ensemble2', ensemble2)
        
        # Evaluate all ensembles first
        self.manager.evaluate_all_ensembles(self.X, self.y, cv_folds=3)
        
        summary = self.manager.get_performance_summary()
        
        # Check summary structure
        self.assertIsInstance(summary, dict)
        self.assertIn('total_ensembles', summary)
        self.assertIn('best_accuracy', summary)
        self.assertIn('worst_accuracy', summary)
        self.assertIn('avg_accuracy', summary)
        
        # Check data types and ranges
        self.assertIsInstance(summary['total_ensembles'], int)
        self.assertGreater(summary['total_ensembles'], 0)
        
        for metric in ['best_accuracy', 'worst_accuracy', 'avg_accuracy']:
            self.assertIsInstance(summary[metric], float)
            self.assertGreaterEqual(summary[metric], 0.0)
            self.assertLessEqual(summary[metric], 1.0)
    
    def test_save_and_load_ensemble(self):
        """Test saving and loading ensemble"""
        ensemble = create_ensemble_model(
            self.base_models, voting_method='soft'
        )
        
        self.manager.add_ensemble('test_ensemble', ensemble)
        
        # Save ensemble
        self.manager.save_ensemble('test_ensemble', 'test_ensemble.pkl')
        
        # Remove ensemble from manager
        self.manager.remove_ensemble('test_ensemble')
        
        # Load ensemble back
        loaded_ensemble = self.manager.load_ensemble('test_ensemble.pkl')
        
        # Check loaded ensemble has required attributes
        self.assertTrue(hasattr(loaded_ensemble, 'fit'))
        self.assertTrue(hasattr(loaded_ensemble, 'predict'))
        self.assertTrue(hasattr(loaded_ensemble, 'predict_proba'))


if __name__ == '__main__':
    unittest.main() 