import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AdvancedConfidenceCalibrator:
    def __init__(self):
        self.calibration_methods = {}
        self.optimal_thresholds = {}
        self.calibration_scores = {}
        
    def platt_scaling(self, model, X_train, y_train, X_val, y_val):
        """Implement Platt scaling for probability calibration"""
        # Get uncalibrated probabilities
        uncalibrated_probs = model.predict_proba(X_val)
        
        # Fit sigmoid function (Platt scaling)
        platt_model = LogisticRegression()
        platt_model.fit(uncalibrated_probs, y_val)
        
        self.calibration_methods['platt'] = platt_model
        
        # Get calibrated probabilities
        calibrated_probs = platt_model.predict_proba(uncalibrated_probs)
        
        return calibrated_probs
    
    def isotonic_regression_calibration(self, model, X_train, y_train, X_val, y_val):
        """Implement isotonic regression for probability calibration"""
        # Get uncalibrated probabilities
        uncalibrated_probs = model.predict_proba(X_val)
        
        # For multi-class, we'll handle each class separately
        calibrated_probs = np.zeros_like(uncalibrated_probs)
        isotonic_models = {}
        
        for class_idx in range(uncalibrated_probs.shape[1]):
            # Create binary labels for this class
            binary_labels = (y_val == class_idx).astype(int)
            
            # Fit isotonic regression
            isotonic_reg = IsotonicRegression(out_of_bounds='clip')
            isotonic_reg.fit(uncalibrated_probs[:, class_idx], binary_labels)
            
            isotonic_models[class_idx] = isotonic_reg
            calibrated_probs[:, class_idx] = isotonic_reg.predict(uncalibrated_probs[:, class_idx])
        
        # Normalize probabilities to sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
        
        self.calibration_methods['isotonic'] = isotonic_models
        
        return calibrated_probs
    
    def temperature_scaling(self, model, X_train, y_train, X_val, y_val):
        """Implement temperature scaling for probability calibration"""
        # Get uncalibrated logits/probabilities
        uncalibrated_probs = model.predict_proba(X_val)
        
        # Convert to logits (inverse softmax)
        epsilon = 1e-15
        logits = np.log(np.clip(uncalibrated_probs, epsilon, 1 - epsilon))
        
        # Find optimal temperature
        def temperature_scaled_probs(logits, temperature):
            scaled_logits = logits / temperature
            # Apply softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        def negative_log_likelihood(temperature):
            calibrated_probs = temperature_scaled_probs(logits, temperature)
            return log_loss(y_val, calibrated_probs)
        
        # Optimize temperature
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(negative_log_likelihood, bounds=(0.1, 10.0), method='bounded')
        optimal_temperature = result.x
        
        self.calibration_methods['temperature'] = optimal_temperature
        
        # Get calibrated probabilities
        calibrated_probs = temperature_scaled_probs(logits, optimal_temperature)
        
        print(f"Optimal temperature: {optimal_temperature:.4f}")
        
        return calibrated_probs
    
    def evaluate_calibration_quality(self, y_true, y_prob_uncalibrated, y_prob_calibrated):
        """Evaluate the quality of calibration using multiple metrics"""
        metrics = {}
        
        # Brier Score (lower is better)
        metrics['brier_score_uncalibrated'] = brier_score_loss(y_true, y_prob_uncalibrated)
        metrics['brier_score_calibrated'] = brier_score_loss(y_true, y_prob_calibrated)
        
        # Log Loss (lower is better)
        metrics['log_loss_uncalibrated'] = log_loss(y_true, y_prob_uncalibrated)
        metrics['log_loss_calibrated'] = log_loss(y_true, y_prob_calibrated)
        
        # Calibration curve analysis
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            y_true, y_prob_uncalibrated, n_bins=10
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_true, y_prob_calibrated, n_bins=10
        )
        
        # Expected Calibration Error (ECE)
        metrics['ece_uncalibrated'] = self._calculate_ece(prob_true_uncal, prob_pred_uncal)
        metrics['ece_calibrated'] = self._calculate_ece(prob_true_cal, prob_pred_cal)
        
        # Maximum Calibration Error (MCE)
        metrics['mce_uncalibrated'] = np.max(np.abs(prob_true_uncal - prob_pred_uncal))
        metrics['mce_calibrated'] = np.max(np.abs(prob_true_cal - prob_pred_cal))
        
        return metrics
    
    def _calculate_ece(self, prob_true, prob_pred):
        """Calculate Expected Calibration Error"""
        return np.mean(np.abs(prob_true - prob_pred))
    
    def find_optimal_thresholds(self, y_true, y_prob, optimization_metric='f1'):
        """Find optimal classification thresholds for each class"""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        optimal_thresholds = {}
        n_classes = y_prob.shape[1]
        
        for class_idx in range(n_classes):
            # Convert to binary classification problem
            binary_true = (y_true == class_idx).astype(int)
            class_probs = y_prob[:, class_idx]
            
            # Test different thresholds
            thresholds = np.linspace(0.1, 0.9, 50)
            best_threshold = 0.5
            best_score = 0
            
            for threshold in thresholds:
                predictions = (class_probs >= threshold).astype(int)
                
                if optimization_metric == 'f1':
                    score = f1_score(binary_true, predictions, zero_division=0)
                elif optimization_metric == 'precision':
                    score = precision_score(binary_true, predictions, zero_division=0)
                elif optimization_metric == 'recall':
                    score = recall_score(binary_true, predictions, zero_division=0)
                else:
                    score = accuracy_score(binary_true, predictions)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            optimal_thresholds[class_idx] = best_threshold
        
        self.optimal_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def calibrate_hierarchical_model(self, hierarchical_predictor, X_val, y_val):
        """Calibrate all levels of hierarchical model"""
        calibrated_models = {}
        
        # Calibrate broad model
        if hasattr(hierarchical_predictor, 'broad_model'):
            broad_calibrated = CalibratedClassifierCV(
                hierarchical_predictor.broad_model, method='isotonic', cv=3
            )
            broad_calibrated.fit(X_val, y_val)  # Use appropriate labels
            calibrated_models['broad'] = broad_calibrated
        
        # Calibrate field model
        if hasattr(hierarchical_predictor, 'field_model'):
            field_calibrated = CalibratedClassifierCV(
                hierarchical_predictor.field_model, method='isotonic', cv=3
            )
            field_calibrated.fit(X_val, y_val)  # Use appropriate labels
            calibrated_models['field'] = field_calibrated
        
        # Calibrate specific model
        if hasattr(hierarchical_predictor, 'specific_model'):
            specific_calibrated = CalibratedClassifierCV(
                hierarchical_predictor.specific_model, method='isotonic', cv=3
            )
            specific_calibrated.fit(X_val, y_val)
            calibrated_models['specific'] = specific_calibrated
        
        return calibrated_models
    
    def plot_calibration_curves(self, y_true, prob_uncalibrated, prob_calibrated, 
                               method_name="Calibrated"):
        """Plot calibration curves for comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot calibration curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            y_true, prob_uncalibrated, n_bins=10
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_true, prob_calibrated, n_bins=10
        )
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(prob_pred_uncal, prob_true_uncal, marker='o', 
                label='Uncalibrated', linewidth=2)
        ax1.plot(prob_pred_cal, prob_true_cal, marker='s', 
                label=method_name, linewidth=2)
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot histogram of probabilities
        ax2.hist(prob_uncalibrated, bins=20, alpha=0.7, label='Uncalibrated', 
                density=True)
        ax2.hist(prob_calibrated, bins=20, alpha=0.7, label=method_name, 
                density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_calibration_analysis(self, model, X_train, y_train, X_test, y_test):
        """Perform comprehensive calibration analysis with multiple methods"""
        print("Performing comprehensive calibration analysis...")
        
        # Split validation data
        X_val, X_eval, y_val, y_eval = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        
        # Get uncalibrated probabilities
        uncalibrated_probs = model.predict_proba(X_eval)
        
        results = {
            'uncalibrated': {
                'probabilities': uncalibrated_probs,
                'predictions': model.predict(X_eval)
            }
        }
        
        # Test different calibration methods
        calibration_methods = ['platt', 'isotonic', 'temperature']
        
        for method in calibration_methods:
            try:
                if method == 'platt':
                    calibrated_probs = self.platt_scaling(model, X_train, y_train, X_eval, y_eval)
                elif method == 'isotonic':
                    calibrated_probs = self.isotonic_regression_calibration(
                        model, X_train, y_train, X_eval, y_eval
                    )
                elif method == 'temperature':
                    calibrated_probs = self.temperature_scaling(
                        model, X_train, y_train, X_eval, y_eval
                    )
                
                results[method] = {
                    'probabilities': calibrated_probs,
                    'predictions': np.argmax(calibrated_probs, axis=1)
                }
                
                # Evaluate calibration quality
                # For binary classification, use positive class probabilities
                if len(np.unique(y_eval)) == 2:
                    uncal_pos_probs = uncalibrated_probs[:, 1]
                    cal_pos_probs = calibrated_probs[:, 1]
                else:
                    # For multi-class, use maximum probability
                    uncal_pos_probs = np.max(uncalibrated_probs, axis=1)
                    cal_pos_probs = np.max(calibrated_probs, axis=1)
                
                # Create binary labels for calibration evaluation
                binary_labels = (y_eval == np.argmax(uncalibrated_probs, axis=1)).astype(int)
                
                metrics = self.evaluate_calibration_quality(
                    binary_labels, uncal_pos_probs, cal_pos_probs
                )
                results[method]['metrics'] = metrics
                
                print(f"\n{method.upper()} Calibration Results:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
            except Exception as e:
                print(f"Error with {method} calibration: {e}")
                continue
        
        # Find best calibration method
        best_method = 'uncalibrated'
        best_score = float('inf')
        
        for method, result in results.items():
            if 'metrics' in result:
                score = result['metrics']['brier_score_calibrated']
                if score < best_score:
                    best_score = score
                    best_method = method
        
        print(f"\nBest calibration method: {best_method}")
        
        return results, best_method

# Integration with existing career predictor
class CalibratedCareerPredictor:
    def __init__(self, base_predictor):
        self.base_predictor = base_predictor
        self.calibrator = AdvancedConfidenceCalibrator()
        self.calibrated_models = {}
        
    def train_with_calibration(self, X_train, y_train, X_val, y_val):
        """Train base model and apply calibration"""
        # Train base model
        self.base_predictor.train_hierarchical_models(X_train, y_train)
        
        # Calibrate models
        self.calibrated_models = self.calibrator.calibrate_hierarchical_model(
            self.base_predictor, X_val, y_val
        )
        
        return self
    
    def predict_with_calibrated_confidence(self, user_features):
        """Make predictions with calibrated confidence"""
        # Use calibrated models if available
        if 'specific' in self.calibrated_models:
            model = self.calibrated_models['specific']
        else:
            model = self.base_predictor.specific_model
        
        # Convert features to appropriate format
        X_user = pd.DataFrame([user_features])
        
        # Get calibrated probabilities
        probabilities = model.predict_proba(X_user)[0]
        prediction = model.predict(X_user)[0]
        
        # Apply optimal thresholds if available
        if hasattr(self.calibrator, 'optimal_thresholds'):
            confidence = probabilities[prediction]
        else:
            confidence = np.max(probabilities)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'calibrated_probabilities': probabilities,
            'is_calibrated': True
        }

# Usage example
if __name__ == "__main__":
    print("Advanced Confidence Calibration System")
    print("Features:")
    print("- Platt scaling")
    print("- Isotonic regression") 
    print("- Temperature scaling")
    print("- Comprehensive calibration analysis")
    print("- Optimal threshold finding")
    print("- Calibration quality evaluation")
