import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from collections import defaultdict
import itertools

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True)
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        self.engineered_feature_names = []
        
    def create_polynomial_features(self, feature_df, max_degree=2):
        """Create polynomial and interaction features"""
        # Select numerical features for polynomial expansion
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            # Fill NaN values before polynomial transformation
            feature_data = feature_df[numerical_cols].fillna(0)
            
            # Create polynomial features (interactions only to avoid explosion)
            poly = PolynomialFeatures(degree=max_degree, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(feature_data)
            
            # Get feature names
            poly_feature_names = poly.get_feature_names_out(numerical_cols)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=feature_df.index)
            
            # Remove original features to avoid duplication
            poly_df = poly_df.drop(columns=numerical_cols, errors='ignore')
            
            return poly_df
        
        return pd.DataFrame(index=feature_df.index)
    
    def create_domain_specific_features(self, academic_grades, user_hobbies, personality_scores):
        """Create domain-specific engineered features"""
        features = {}
        
        # Academic performance patterns
        stem_subjects = ['mathematics', 'science']
        humanities_subjects = ['english', 'social_science', 'second_language']
        
        stem_scores = [academic_grades.get(s, 0.5) for s in stem_subjects]
        humanities_scores = [academic_grades.get(s, 0.5) for s in humanities_subjects]
        
        # Advanced academic features
        features['academic_stem_dominance'] = np.mean(stem_scores) - np.mean(humanities_scores)
        features['academic_variance'] = np.var(list(academic_grades.values()))
        features['academic_skewness'] = self._calculate_skewness(list(academic_grades.values()))
        features['academic_peak_performance'] = max(academic_grades.values()) if academic_grades else 0.5
        features['academic_floor_performance'] = min(academic_grades.values()) if academic_grades else 0.5
        features['academic_range'] = features['academic_peak_performance'] - features['academic_floor_performance']
        
        # Hobby intensity patterns
        hobby_intensities = [h.get('intensity', 0) for h in user_hobbies.values()]
        if hobby_intensities:
            features['hobby_intensity_mean'] = np.mean(hobby_intensities)
            features['hobby_intensity_std'] = np.std(hobby_intensities)
            features['hobby_intensity_max'] = max(hobby_intensities)
            features['hobby_focus_ratio'] = max(hobby_intensities) / (np.mean(hobby_intensities) + 1e-6)
        else:
            features.update({
                'hobby_intensity_mean': 0, 'hobby_intensity_std': 0,
                'hobby_intensity_max': 0, 'hobby_focus_ratio': 0
            })
        
        # Hobby experience patterns
        hobby_years = [h.get('years', 0) for h in user_hobbies.values()]
        if hobby_years:
            features['hobby_experience_total'] = sum(hobby_years)
            features['hobby_experience_avg'] = np.mean(hobby_years)
            features['hobby_experience_max'] = max(hobby_years)
            features['hobby_commitment_consistency'] = 1 - (np.std(hobby_years) / (np.mean(hobby_years) + 1e-6))
        else:
            features.update({
                'hobby_experience_total': 0, 'hobby_experience_avg': 0,
                'hobby_experience_max': 0, 'hobby_commitment_consistency': 0
            })
        
        # Personality trait combinations
        big_five_scores = list(personality_scores.values())
        features['personality_balance'] = 1 - np.std(big_five_scores)
        features['personality_extremity'] = max([abs(s - 0.5) for s in big_five_scores])
        
        # Career-relevant personality combinations
        features['creative_analytical_balance'] = (
            personality_scores.get('openness', 0.5) * 
            personality_scores.get('conscientiousness', 0.5)
        )
        
        features['social_leadership_potential'] = (
            personality_scores.get('extraversion', 0.5) * 0.4 +
            personality_scores.get('agreeableness', 0.5) * 0.3 +
            personality_scores.get('conscientiousness', 0.5) * 0.3
        )
        
        features['stress_resilience'] = (
            (1 - personality_scores.get('neuroticism', 0.5)) * 0.6 +
            personality_scores.get('conscientiousness', 0.5) * 0.4
        )
        
        return features
    
    def create_temporal_features(self, user_hobbies):
        """Create features based on temporal patterns in hobbies"""
        features = {}
        
        if not user_hobbies:
            return {'temporal_consistency': 0, 'temporal_trend': 0, 'temporal_diversity': 0}
        
        # Sort hobbies by years of experience
        sorted_hobbies = sorted(user_hobbies.items(), key=lambda x: x[1].get('years', 0))
        
        # Temporal consistency (do newer hobbies have higher intensity?)
        years = [h[1].get('years', 0) for h in sorted_hobbies]
        intensities = [h[1].get('intensity', 0) for h in sorted_hobbies]
        
        if len(years) > 1:
            # Calculate correlation between years and intensity
            try:
                corr_coef = np.corrcoef(years, intensities)[0, 1] if len(set(years)) > 1 else 0
                features['temporal_consistency'] = 0 if np.isnan(corr_coef) else corr_coef
            except:
                features['temporal_consistency'] = 0
        else:
            features['temporal_consistency'] = 0
        
        # Temporal diversity (how spread out are the hobby starting times?)
        features['temporal_diversity'] = np.std(years) if len(years) > 1 else 0
        
        # Recent hobby trend (are recent hobbies more intense?)
        recent_hobbies = [h for h in user_hobbies.values() if h.get('years', 0) <= 2]
        old_hobbies = [h for h in user_hobbies.values() if h.get('years', 0) > 2]
        
        if recent_hobbies and old_hobbies:
            recent_intensity = np.mean([h.get('intensity', 0) for h in recent_hobbies])
            old_intensity = np.mean([h.get('intensity', 0) for h in old_hobbies])
            features['temporal_trend'] = recent_intensity - old_intensity
        else:
            features['temporal_trend'] = 0
        
        return features
    
    def create_clustering_features(self, feature_df, n_clusters=5):
        """Create features based on clustering of similar profiles"""
        from sklearn.cluster import KMeans
        
        if len(feature_df) < n_clusters:
            return pd.DataFrame(index=feature_df.index)
        
        # Perform clustering on normalized features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df.fillna(0))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        cluster_df = pd.DataFrame(index=feature_df.index)
        
        # Add cluster membership as features
        for i in range(n_clusters):
            cluster_df[f'cluster_{i}'] = (clusters == i).astype(int)
        
        # Add distance to each cluster center
        distances = kmeans.transform(scaled_features)
        for i in range(n_clusters):
            cluster_df[f'distance_to_cluster_{i}'] = distances[:, i]
        
        return cluster_df
    
    def select_important_features(self, X, y, k=50):
        """Feature selection using multiple methods"""
        # Method 1: Mutual Information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        mi_scores = mi_selector.fit(X, y).scores_
        
        # Method 2: F-statistic
        f_selector = SelectKBest(score_func=f_classif, k=k)
        f_scores = f_selector.fit(X, y).scores_
        
        # Combine scores (normalized)
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-6)
        f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-6)
        
        combined_scores = 0.6 * mi_scores_norm + 0.4 * f_scores_norm
        
        # Select top k features
        top_k_indices = np.argsort(combined_scores)[-k:]
        
        return X.iloc[:, top_k_indices], top_k_indices
    
    def engineer_all_features_advanced(self, academic_grades, user_hobbies, personality_scores):
        """Advanced feature engineering pipeline"""
        # Basic features (from original implementation)
        from src.feature_engineering import AdvancedFeatureEngineer as BaseEngineer
        base_engineer = BaseEngineer()
        base_features = base_engineer.engineer_all_features(academic_grades, user_hobbies, personality_scores)
        
        # Domain-specific features
        domain_features = self.create_domain_specific_features(academic_grades, user_hobbies, personality_scores)
        
        # Temporal features
        temporal_features = self.create_temporal_features(user_hobbies)
        
        # Combine all basic features
        all_features = {**base_features, **domain_features, **temporal_features}
        feature_df = pd.DataFrame([all_features])
        
        # Polynomial features
        poly_features = self.create_polynomial_features(feature_df, max_degree=2)
        
        # Combine everything
        final_df = pd.concat([feature_df, poly_features], axis=1)
        
        # Fill any remaining NaN values
        final_df = final_df.fillna(0)
        
        return final_df.iloc[0].to_dict()
    
    def _calculate_skewness(self, values):
        """Calculate skewness of a distribution"""
        if len(values) < 3:
            return 0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0
        
        skewness = np.mean([((x - mean_val) / std_val) ** 3 for x in values])
        return skewness

# Integration with existing pipeline
def enhance_existing_feature_engineering():
    """Example of how to integrate with existing code"""
    
    class EnhancedFeatureEngineer(AdvancedFeatureEngineer):
        def __init__(self):
            super().__init__()
            # Import original engineer
            from src.feature_engineering import AdvancedFeatureEngineer as OriginalEngineer
            self.original_engineer = OriginalEngineer()
        
        def engineer_all_features(self, academic_grades, user_hobbies, personality_scores):
            """Enhanced version of the original method"""
            return self.engineer_all_features_advanced(academic_grades, user_hobbies, personality_scores)
    
    return EnhancedFeatureEngineer
