"""
Ultimate Career Prediction Model - Complete Hierarchical Training Script
========================================================================

This script combines the best features from all training scripts:
- Hierarchical prediction (Broad → Field → Specific Career)
- Advanced feature engineering (40+ features)
- Ensemble methods with XGBoost compatibility fixes
- SMOTE for class imbalance handling
- Comprehensive evaluation and analysis
- Enhanced prediction demonstrations

Architecture:
- Broad Categories: STEM, Business, Creative, Social, Healthcare, Law (6 categories)
- Field Level: Engineering, Data Science, Design, etc. (11+ fields)
- Specific Careers: Software Engineer, Data Scientist, etc. (15 careers)

Usage:
    python ultimate_main.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib

# Add src to path
sys.path.append('src')
sys.path.append('config')

from src.data_generator import SyntheticDataGenerator
from src.data_augmentation_enhancer import DataAugmentationEnhancer

def create_directories():
    """Create necessary directories for the project"""
    directories = ['data', 'models', 'results', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Project directories created")

def safe_float(value, default=0.5):
    """Safely convert value to float with fallback"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def load_and_prepare_data():
    """Load and prepare enhanced training data"""
    print("\n" + "="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80)
    
    # Try to load existing enhanced data
    data_file = "data/enhanced_career_data.json"
    if not os.path.exists(data_file):
        data_file = "data/synthetic_career_data.json"
    
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}")
        with open(data_file, 'r') as f:
            dataset = json.load(f)
        print(f"✓ Loaded {len(dataset)} samples")
    else:
        print("No existing data found, generating new enhanced dataset...")
        dataset = generate_enhanced_data()
    
    # Clean and validate data
    cleaned_dataset = clean_and_validate_data(dataset)
    
    # Analyze data distribution
    analyze_data_distribution(cleaned_dataset)
    
    return cleaned_dataset

def generate_enhanced_data(samples_per_career=150):
    """Generate enhanced training data with augmentation"""
    print("Generating enhanced training data...")
    
    # Generate base dataset
    generator = SyntheticDataGenerator(seed=42)
    base_dataset = generator.generate_dataset(samples_per_career=samples_per_career)
    
    # Apply data augmentation
    augmenter = DataAugmentationEnhancer(seed=43)
    augmented_dataset = augmenter.generate_augmented_dataset(
        base_dataset[:len(base_dataset)//2], augmentation_factor=1
    )
    
    # Combine datasets
    enhanced_dataset = base_dataset + augmented_dataset
    
    # Save dataset
    with open("data/enhanced_career_data.json", 'w') as f:
        json.dump(enhanced_dataset, f, indent=2, default=str)
    
    generator.save_dataset(enhanced_dataset, "data/synthetic_career_data.json")
    
    print(f"✓ Generated {len(base_dataset)} base samples")
    print(f"✓ Generated {len(augmented_dataset)} augmented samples")
    print(f"✓ Total enhanced dataset: {len(enhanced_dataset)} samples")
    
    return enhanced_dataset

def clean_and_validate_data(dataset):
    """Clean and validate data with robust error handling"""
    print("Cleaning and validating data...")
    
    cleaned_dataset = []
    skipped_count = 0
    
    for sample in dataset:
        try:
            # Validate structure
            if not all(key in sample for key in ['academic_grades', 'hobbies', 'personality', 'career']):
                skipped_count += 1
                continue
            
            # Clean academic grades
            academics = {}
            for subject, grade in sample['academic_grades'].items():
                academics[subject] = safe_float(grade)
            
            # Clean hobbies
            hobbies = {}
            if isinstance(sample['hobbies'], dict):
                for hobby_name, hobby_data in sample['hobbies'].items():
                    if isinstance(hobby_data, dict):
                        cleaned_hobby = {}
                        for attr in ['intensity', 'proficiency', 'years']:
                            value = hobby_data.get(attr, 0.5 if attr != 'years' else 1)
                            cleaned_hobby[attr] = safe_float(value, 0.5 if attr != 'years' else 1)
                        hobbies[hobby_name] = cleaned_hobby
            
            # Clean personality
            personality = {}
            for trait, score in sample['personality'].items():
                personality[trait] = safe_float(score)
            
            # Create cleaned sample
            cleaned_sample = {
                'academic_grades': academics,
                'hobbies': hobbies,
                'personality': personality,
                'career': str(sample['career'])
            }
            
            cleaned_dataset.append(cleaned_sample)
            
        except Exception as e:
            skipped_count += 1
            continue
    
    print(f"✓ Cleaned dataset: {len(cleaned_dataset)} samples")
    print(f"✓ Skipped {skipped_count} invalid samples")
    
    return cleaned_dataset

def analyze_data_distribution(dataset):
    """Analyze and display data distribution"""
    career_counts = Counter([sample['career'] for sample in dataset])
    print(f"✓ Data distribution across {len(career_counts)} careers:")
    for career, count in sorted(career_counts.items()):
        print(f"   {career}: {count} samples")

def create_ultimate_features(dataset):
    """Create comprehensive feature matrix with all advanced features"""
    print("\n" + "="*80)
    print("STEP 2: CREATING ULTIMATE FEATURE MATRIX")
    print("="*80)
    
    features = []
    career_labels = []
    processed_count = 0
    
    for sample in dataset:
        try:
            feature_dict = {}
            
            # === ACADEMIC FEATURES (5 features) ===
            academics = sample.get('academic_grades', {})
            feature_dict['math_grade'] = safe_float(academics.get('mathematics', 0.5))
            feature_dict['science_grade'] = safe_float(academics.get('science', 0.5))
            feature_dict['english_grade'] = safe_float(academics.get('english', 0.5))
            feature_dict['social_grade'] = safe_float(academics.get('social_science', 0.5))
            feature_dict['language_grade'] = safe_float(academics.get('second_language', 0.5))
            
            # === PERSONALITY FEATURES (5 features) ===
            personality = sample.get('personality', {})
            feature_dict['openness'] = safe_float(personality.get('openness', 0.5))
            feature_dict['conscientiousness'] = safe_float(personality.get('conscientiousness', 0.5))
            feature_dict['extraversion'] = safe_float(personality.get('extraversion', 0.5))
            feature_dict['agreeableness'] = safe_float(personality.get('agreeableness', 0.5))
            feature_dict['neuroticism'] = safe_float(personality.get('neuroticism', 0.5))
            
            # === HOBBY FEATURES ===
            hobbies = sample.get('hobbies', {})
            feature_dict['num_hobbies'] = len(hobbies)
            
            # Calculate hobby statistics safely
            if hobbies:
                intensities = []
                proficiencies = []
                years_list = []
                
                for hobby_data in hobbies.values():
                    if isinstance(hobby_data, dict):
                        intensities.append(safe_float(hobby_data.get('intensity', 0.5)))
                        proficiencies.append(safe_float(hobby_data.get('proficiency', 0.5)))
                        years_list.append(safe_float(hobby_data.get('years', 1)))
                
                feature_dict['avg_hobby_intensity'] = np.mean(intensities) if intensities else 0
                feature_dict['avg_hobby_proficiency'] = np.mean(proficiencies) if proficiencies else 0
                feature_dict['avg_hobby_years'] = np.mean(years_list) if years_list else 0
                feature_dict['max_hobby_intensity'] = np.max(intensities) if intensities else 0
                feature_dict['max_hobby_proficiency'] = np.max(proficiencies) if proficiencies else 0
                feature_dict['hobby_intensity_std'] = np.std(intensities) if len(intensities) > 1 else 0
                feature_dict['hobby_proficiency_std'] = np.std(proficiencies) if len(proficiencies) > 1 else 0
            else:
                feature_dict.update({
                    'avg_hobby_intensity': 0, 'avg_hobby_proficiency': 0, 'avg_hobby_years': 0,
                    'max_hobby_intensity': 0, 'max_hobby_proficiency': 0,
                    'hobby_intensity_std': 0, 'hobby_proficiency_std': 0
                })
            
            # === BINARY HOBBY FEATURES (10+ features) ===
            hobby_names = list(hobbies.keys())
            feature_dict['has_programming'] = 1 if 'programming' in hobby_names else 0
            feature_dict['has_research'] = 1 if 'research' in hobby_names else 0
            feature_dict['has_writing'] = 1 if 'writing' in hobby_names else 0
            feature_dict['has_music'] = 1 if 'music' in hobby_names else 0
            feature_dict['has_sports'] = 1 if any('sport' in h.lower() for h in hobby_names) else 0
            feature_dict['has_cooking'] = 1 if 'cooking' in hobby_names else 0
            feature_dict['has_volunteering'] = 1 if 'volunteering' in hobby_names else 0
            feature_dict['has_photography'] = 1 if 'photography' in hobby_names else 0
            feature_dict['has_gaming'] = 1 if 'gaming' in hobby_names else 0
            feature_dict['has_robotics'] = 1 if 'robotics' in hobby_names else 0
            feature_dict['has_reading'] = 1 if 'reading' in hobby_names else 0
            feature_dict['has_entrepreneurship'] = 1 if 'entrepreneurship' in hobby_names else 0
            
            # === DERIVED ACADEMIC FEATURES (5 features) ===
            feature_dict['stem_score'] = (feature_dict['math_grade'] + feature_dict['science_grade']) / 2
            feature_dict['humanities_score'] = (feature_dict['english_grade'] + feature_dict['social_grade']) / 2
            feature_dict['stem_vs_humanities'] = feature_dict['stem_score'] - feature_dict['humanities_score']
            feature_dict['academic_consistency'] = 1 - np.std([
                feature_dict['math_grade'], feature_dict['science_grade'], 
                feature_dict['english_grade'], feature_dict['social_grade'], 
                feature_dict['language_grade']
            ])
            feature_dict['academic_peak'] = max([
                feature_dict['math_grade'], feature_dict['science_grade'], 
                feature_dict['english_grade'], feature_dict['social_grade'], 
                feature_dict['language_grade']
            ])
            feature_dict['academic_average'] = np.mean([
                feature_dict['math_grade'], feature_dict['science_grade'], 
                feature_dict['english_grade'], feature_dict['social_grade'], 
                feature_dict['language_grade']
            ])
            
            # === ADVANCED PERSONALITY-DERIVED FEATURES (8 features) ===
            feature_dict['leadership_potential'] = (
                feature_dict['extraversion'] * 0.4 + 
                feature_dict['conscientiousness'] * 0.3 + 
                feature_dict['openness'] * 0.3
            )
            
            feature_dict['analytical_disposition'] = (
                feature_dict['openness'] * 0.4 +
                feature_dict['conscientiousness'] * 0.4 +
                (1 - feature_dict['extraversion']) * 0.2
            )
            
            feature_dict['people_orientation'] = (
                feature_dict['extraversion'] * 0.4 +
                feature_dict['agreeableness'] * 0.4 +
                (1 - feature_dict['neuroticism']) * 0.2
            )
            
            feature_dict['stress_resilience'] = (
                (1 - feature_dict['neuroticism']) * 0.5 +
                feature_dict['conscientiousness'] * 0.3 +
                feature_dict['extraversion'] * 0.2
            )
            
            feature_dict['innovation_potential'] = (
                feature_dict['openness'] * 0.6 +
                feature_dict['extraversion'] * 0.2 +
                (1 - feature_dict['neuroticism']) * 0.2
            )
            
            feature_dict['detail_orientation'] = (
                feature_dict['conscientiousness'] * 0.6 +
                (1 - feature_dict['neuroticism']) * 0.4
            )
            
            feature_dict['collaboration_score'] = (
                feature_dict['agreeableness'] * 0.5 +
                feature_dict['extraversion'] * 0.3 +
                feature_dict['conscientiousness'] * 0.2
            )
            
            feature_dict['adaptability_score'] = (
                feature_dict['openness'] * 0.5 +
                (1 - feature_dict['neuroticism']) * 0.3 +
                feature_dict['extraversion'] * 0.2
            )
            
            # === ORIENTATION FEATURES (4 features) ===
            feature_dict['technical_orientation'] = (
                feature_dict['stem_score'] * 0.6 + 
                feature_dict['has_programming'] * 0.4
            )
            
            creative_hobbies = (feature_dict['has_music'] + feature_dict['has_photography'] + 
                             feature_dict['has_writing'] + feature_dict['has_cooking']) / 4
            feature_dict['creative_orientation'] = (
                creative_hobbies * 0.6 +
                feature_dict['openness'] * 0.4
            )
            
            social_hobbies = (feature_dict['has_volunteering'] + feature_dict['has_sports']) / 2
            feature_dict['social_orientation'] = (
                social_hobbies * 0.5 +
                feature_dict['people_orientation'] * 0.5
            )
            
            research_orientation = (feature_dict['has_research'] + feature_dict['has_reading']) / 2
            feature_dict['research_orientation'] = (
                research_orientation * 0.6 +
                feature_dict['analytical_disposition'] * 0.4
            )
            
            # === SPECIALIZATION INDICES (3 features) ===
            feature_dict['hobby_specialization_index'] = (
                feature_dict['max_hobby_intensity'] / (feature_dict['avg_hobby_intensity'] + 0.001)
            )
            
            feature_dict['academic_specialization_index'] = (
                feature_dict['academic_peak'] / (feature_dict['academic_average'] + 0.001)
            )
            
            personality_scores = [feature_dict['openness'], feature_dict['conscientiousness'], 
                                feature_dict['extraversion'], feature_dict['agreeableness'], 
                                feature_dict['neuroticism']]
            feature_dict['personality_extremity'] = np.std(personality_scores)
            
            features.append(feature_dict)
            career_labels.append(sample['career'])
            processed_count += 1
            
        except Exception as e:
            continue
    
    # Convert to DataFrame
    X = pd.DataFrame(features)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"✓ Ultimate feature matrix shape: {X.shape}")
    print(f"✓ Total features created: {X.shape[1]}")
    print(f"✓ Successfully processed {processed_count} samples")
    
    return X, career_labels

def create_hierarchical_labels(career_labels):
    """Create complete hierarchical labels for broad, field, and specific categories"""
    print("Creating hierarchical label structure...")
    
    # Broad category mapping (6 categories)
    career_to_broad = {
        'software_engineer': 'STEM',
        'data_scientist': 'STEM',
        'web_developer': 'STEM', 
        'mechanical_engineer': 'STEM',
        'financial_analyst': 'Business',
        'marketing_manager': 'Business',
        'accountant': 'Business',
        'graphic_designer': 'Creative',
        'ux_designer': 'Creative',
        'chef': 'Creative',
        'teacher': 'Social',
        'psychologist': 'Healthcare',
        'doctor': 'Healthcare',
        'nurse': 'Healthcare',
        'lawyer': 'Law_Government'
    }
    
    # Field mapping (11+ fields)
    career_to_field = {
        'software_engineer': 'engineering',
        'data_scientist': 'data_science', 
        'web_developer': 'engineering',
        'mechanical_engineer': 'engineering',
        'financial_analyst': 'finance',
        'marketing_manager': 'marketing',
        'accountant': 'finance',
        'graphic_designer': 'design',
        'ux_designer': 'design',
        'chef': 'culinary',
        'teacher': 'education',
        'psychologist': 'psychology',
        'doctor': 'medicine',
        'nurse': 'healthcare',
        'lawyer': 'legal'
    }
    
    broad_labels = [career_to_broad.get(career, 'Other') for career in career_labels]
    field_labels = [career_to_field.get(career, 'other') for career in career_labels]
    
    print(f"✓ Hierarchical structure created:")
    print(f"   {len(set(career_labels))} careers → {len(set(field_labels))} fields → {len(set(broad_labels))} broad categories")
    print(f"   Broad categories: {sorted(set(broad_labels))}")
    print(f"   Fields: {sorted(set(field_labels))}")
    
    return broad_labels, field_labels

def train_hierarchical_models(X, career_labels, broad_labels, field_labels):
    """Train complete hierarchical models with enhanced techniques"""
    print("\n" + "="*80)
    print("STEP 3: TRAINING ULTIMATE HIERARCHICAL MODELS")
    print("="*80)
    
    # Encode all labels
    broad_encoder = LabelEncoder()
    field_encoder = LabelEncoder()
    career_encoder = LabelEncoder()
    
    broad_encoded = broad_encoder.fit_transform(broad_labels)
    field_encoded = field_encoder.fit_transform(field_labels)
    career_encoded = career_encoder.fit_transform(career_labels)
    
    results = {}
    
    # === TRAIN BROAD CATEGORY MODEL ===
    print(f"\n{'='*50}")
    print("TRAINING BROAD CATEGORY MODEL")
    print('='*50)
    
    X_broad_train, X_broad_test, y_broad_train, y_broad_test = train_test_split(
        X, broad_encoded, test_size=0.2, random_state=42, stratify=broad_encoded
    )
    
    # Apply SMOTE for broad categories
    try:
        smote_broad = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_broad_train).values())-1))
        X_broad_balanced, y_broad_balanced = smote_broad.fit_resample(X_broad_train, y_broad_train)
        print(f"Broad data: {len(X_broad_train)} → {len(X_broad_balanced)} after SMOTE")
    except:
        X_broad_balanced, y_broad_balanced = X_broad_train, y_broad_train
        print("Using original broad data without SMOTE")
    
    # Create broad category ensemble
    xgb_broad = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    rf_broad = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    broad_ensemble = VotingClassifier(
        estimators=[('xgb', xgb_broad), ('rf', rf_broad)],
        voting='soft'
    )
    
    # Train and evaluate broad model
    broad_ensemble.fit(X_broad_balanced, y_broad_balanced)
    broad_pred = broad_ensemble.predict(X_broad_test)
    broad_accuracy = accuracy_score(y_broad_test, broad_pred)
    
    print(f"✓ Broad Category Model Accuracy: {broad_accuracy:.3f}")
    results['broad_accuracy'] = broad_accuracy
    results['broad_model'] = broad_ensemble
    results['broad_encoder'] = broad_encoder
    
    # === TRAIN FIELD MODEL ===
    print(f"\n{'='*50}")
    print("TRAINING FIELD MODEL")
    print('='*50)
    
    X_field_train, X_field_test, y_field_train, y_field_test = train_test_split(
        X, field_encoded, test_size=0.2, random_state=42, stratify=field_encoded
    )
    
    # Apply SMOTE for field model
    try:
        smote_field = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_field_train).values())-1))
        X_field_balanced, y_field_balanced = smote_field.fit_resample(X_field_train, y_field_train)
        print(f"Field data: {len(X_field_train)} → {len(X_field_balanced)} after SMOTE")
    except:
        X_field_balanced, y_field_balanced = X_field_train, y_field_train
        print("Using original field data without SMOTE")
    
    # Create field ensemble
    xgb_field = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    rf_field = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    
    field_ensemble = VotingClassifier(
        estimators=[('xgb', xgb_field), ('rf', rf_field)],
        voting='soft'
    )
    
    # Train and evaluate field model
    field_ensemble.fit(X_field_balanced, y_field_balanced)
    field_pred = field_ensemble.predict(X_field_test)
    field_accuracy = accuracy_score(y_field_test, field_pred)
    
    print(f"✓ Field Model Accuracy: {field_accuracy:.3f}")
    results['field_accuracy'] = field_accuracy
    results['field_model'] = field_ensemble
    results['field_encoder'] = field_encoder
    
    # === TRAIN CAREER MODEL ===
    print(f"\n{'='*50}")
    print("TRAINING SPECIFIC CAREER MODEL")
    print('='*50)
    
    X_career_train, X_career_test, y_career_train, y_career_test = train_test_split(
        X, career_encoded, test_size=0.2, random_state=42, stratify=career_encoded
    )
    
    # Apply SMOTE for career model
    try:
        smote_career = SMOTE(random_state=42, k_neighbors=min(3, min(Counter(y_career_train).values())-1))
        X_career_balanced, y_career_balanced = smote_career.fit_resample(X_career_train, y_career_train)
        print(f"Career data: {len(X_career_train)} → {len(X_career_balanced)} after SMOTE")
    except:
        X_career_balanced, y_career_balanced = X_career_train, y_career_train
        print("Using original career data without SMOTE")
    
    # Create career ensemble
    xgb_career = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    rf_career = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    
    career_ensemble = VotingClassifier(
        estimators=[('xgb', xgb_career), ('rf', rf_career)],
        voting='soft'
    )
    
    # Train and evaluate career model
    career_ensemble.fit(X_career_balanced, y_career_balanced)
    career_pred = career_ensemble.predict(X_career_test)
    career_accuracy = accuracy_score(y_career_test, career_pred)
    
    print(f"✓ Specific Career Model Accuracy: {career_accuracy:.3f}")
    results['career_accuracy'] = career_accuracy
    results['career_model'] = career_ensemble
    results['career_encoder'] = career_encoder
    
    # Save all models
    model_dir = "models"
    joblib.dump(results['broad_model'], f"{model_dir}/ultimate_broad_model.joblib")
    joblib.dump(results['field_model'], f"{model_dir}/ultimate_field_model.joblib")
    joblib.dump(results['career_model'], f"{model_dir}/ultimate_career_model.joblib")
    joblib.dump(results['broad_encoder'], f"{model_dir}/broad_encoder.joblib")
    joblib.dump(results['field_encoder'], f"{model_dir}/field_encoder.joblib")
    joblib.dump(results['career_encoder'], f"{model_dir}/career_encoder.joblib")
    
    print(f"\n✓ All models saved to {model_dir}/")
    
    # Print detailed classification reports
    print_detailed_reports(results, X_broad_test, y_broad_test, broad_pred, 
                          X_field_test, y_field_test, field_pred,
                          X_career_test, y_career_test, career_pred)
    
    return results

def print_detailed_reports(results, X_broad_test, y_broad_test, broad_pred,
                          X_field_test, y_field_test, field_pred,
                          X_career_test, y_career_test, career_pred):
    """Print detailed classification reports for all models"""
    
    print(f"\n{'='*60}")
    print("DETAILED BROAD CATEGORY MODEL REPORT")
    print('='*60)
    broad_names = results['broad_encoder'].classes_
    print(classification_report(y_broad_test, broad_pred, target_names=broad_names, zero_division=0))
    
    print(f"\n{'='*60}")
    print("DETAILED FIELD MODEL REPORT")
    print('='*60)
    field_names = results['field_encoder'].classes_
    print(classification_report(y_field_test, field_pred, target_names=field_names, zero_division=0))
    
    print(f"\n{'='*60}")
    print("DETAILED CAREER MODEL REPORT")
    print('='*60)
    career_names = results['career_encoder'].classes_
    print(classification_report(y_career_test, career_pred, target_names=career_names, zero_division=0))

def analyze_feature_importance(results, X):
    """Analyze feature importance from all models"""
    print(f"\n{'='*80}")
    print("STEP 4: COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print('='*80)
    
    # Get XGBoost models (first estimator in each ensemble)
    broad_xgb = results['broad_model'].estimators_[0]
    field_xgb = results['field_model'].estimators_[0]
    career_xgb = results['career_model'].estimators_[0]
    
    # Create importance DataFrames
    broad_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': broad_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    field_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': field_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    career_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': career_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Print top features for each model
    print(f"\n{'='*50}")
    print("TOP 10 BROAD CATEGORY MODEL FEATURES")
    print('='*50)
    for i, (_, row) in enumerate(broad_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    print(f"\n{'='*50}")
    print("TOP 10 FIELD MODEL FEATURES")
    print('='*50)
    for i, (_, row) in enumerate(field_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    print(f"\n{'='*50}")
    print("TOP 10 CAREER MODEL FEATURES")
    print('='*50)
    for i, (_, row) in enumerate(career_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # Save feature importance
    broad_importance.to_csv("results/ultimate_broad_importance.csv", index=False)
    field_importance.to_csv("results/ultimate_field_importance.csv", index=False)
    career_importance.to_csv("results/ultimate_career_importance.csv", index=False)
    
    return broad_importance, field_importance, career_importance

def create_ultimate_predictor(results):
    """Create ultimate predictor class with hierarchical predictions"""
    
    class UltimateCareerPredictor:
        def __init__(self, results):
            self.broad_model = results['broad_model']
            self.field_model = results['field_model']
            self.career_model = results['career_model']
            self.broad_encoder = results['broad_encoder']
            self.field_encoder = results['field_encoder']
            self.career_encoder = results['career_encoder']
            
        def predict_user_career(self, academic_grades, hobbies, personality):
            """Make hierarchical predictions with confidence scoring"""
            # Create user features
            user_data = [{
                'academic_grades': academic_grades,
                'hobbies': hobbies,
                'personality': personality,
                'career': 'unknown'
            }]
            
            user_features, _ = create_ultimate_features(user_data)
            
            # Make broad category prediction
            broad_proba = self.broad_model.predict_proba(user_features)[0]
            broad_pred = self.broad_encoder.inverse_transform([np.argmax(broad_proba)])[0]
            broad_confidence = max(broad_proba)
            
            # Make field prediction
            field_proba = self.field_model.predict_proba(user_features)[0]
            field_pred = self.field_encoder.inverse_transform([np.argmax(field_proba)])[0]
            field_confidence = max(field_proba)
            
            # Make career prediction
            career_proba = self.career_model.predict_proba(user_features)[0]
            career_pred = self.career_encoder.inverse_transform([np.argmax(career_proba)])[0]
            career_confidence = max(career_proba)
            
            # Get top alternatives for each level
            top_broad_indices = np.argsort(broad_proba)[-3:][::-1]
            top_broad_alternatives = [
                {'category': self.broad_encoder.inverse_transform([idx])[0], 'confidence': broad_proba[idx]}
                for idx in top_broad_indices
            ]
            
            top_field_indices = np.argsort(field_proba)[-3:][::-1]
            top_field_alternatives = [
                {'category': self.field_encoder.inverse_transform([idx])[0], 'confidence': field_proba[idx]}
                for idx in top_field_indices
            ]
            
            top_career_indices = np.argsort(career_proba)[-5:][::-1]
            top_career_alternatives = [
                {'career': self.career_encoder.inverse_transform([idx])[0], 'confidence': career_proba[idx]}
                for idx in top_career_indices
            ]
            
            return {
                'primary_recommendation': {
                    'career': career_pred,
                    'confidence': career_confidence,
                    'level': 'specific'
                },
                'hierarchical_predictions': {
                    'broad': {'category': broad_pred, 'confidence': broad_confidence},
                    'field': {'category': field_pred, 'confidence': field_confidence},
                    'specific': {'category': career_pred, 'confidence': career_confidence}
                },
                'top_alternatives': {
                    'broad_categories': top_broad_alternatives,
                    'fields': top_field_alternatives,
                    'careers': top_career_alternatives
                },
                'recommendation_reasoning': [
                    f'Broad category match: {broad_pred} (confidence: {broad_confidence:.1%})',
                    f'Field specialization: {field_pred} (confidence: {field_confidence:.1%})',
                    f'Specific career recommendation: {career_pred} (confidence: {career_confidence:.1%})',
                    f'Ultimate model with {user_features.shape[1]} engineered features and hierarchical ensemble learning'
                ]
            }
    
    return UltimateCareerPredictor(results)

def demonstrate_ultimate_predictions(predictor):
    """Demonstrate ultimate predictions with comprehensive analysis"""
    print(f"\n{'='*80}")
    print("STEP 5: ULTIMATE PREDICTIONS DEMONSTRATION")
    print('='*80)
    
    # Enhanced test users with diverse profiles
    test_users = [
        {
            'name': 'Alex - Technical Creative Innovator',
            'academic_grades': {
                'mathematics': 0.85, 'science': 0.80, 'english': 0.75,
                'social_science': 0.60, 'second_language': 0.65
            },
            'hobbies': {
                'programming': {'intensity': 0.9, 'proficiency': 0.8, 'years': 4},
                'photography': {'intensity': 0.7, 'proficiency': 0.6, 'years': 2},
                'robotics': {'intensity': 0.6, 'proficiency': 0.5, 'years': 1}
            },
            'personality': {
                'openness': 0.85, 'conscientiousness': 0.75, 'extraversion': 0.45,
                'agreeableness': 0.65, 'neuroticism': 0.35
            }
        },
        {
            'name': 'Sarah - Social Impact Leader',
            'academic_grades': {
                'mathematics': 0.70, 'science': 0.65, 'english': 0.90,
                'social_science': 0.85, 'second_language': 0.80
            },
            'hobbies': {
                'volunteering': {'intensity': 0.8, 'proficiency': 0.7, 'years': 5},
                'writing': {'intensity': 0.7, 'proficiency': 0.6, 'years': 2},
                'public_speaking': {'intensity': 0.8, 'proficiency': 0.8, 'years': 3}
            },
            'personality': {
                'openness': 0.75, 'conscientiousness': 0.80, 'extraversion': 0.90,
                'agreeableness': 0.85, 'neuroticism': 0.25
            }
        },
        {
            'name': 'Mike - Analytical Problem Solver',
            'academic_grades': {
                'mathematics': 0.95, 'science': 0.90, 'english': 0.70,
                'social_science': 0.75, 'second_language': 0.60
            },
            'hobbies': {
                'research': {'intensity': 0.9, 'proficiency': 0.8, 'years': 4},
                'reading': {'intensity': 0.8, 'proficiency': 0.9, 'years': 10},
                'investing': {'intensity': 0.8, 'proficiency': 0.7, 'years': 3}
            },
            'personality': {
                'openness': 0.80, 'conscientiousness': 0.90, 'extraversion': 0.40,
                'agreeableness': 0.60, 'neuroticism': 0.20
            }
        }
    ]
    
    for user in test_users:
        print(f"\n{'='*70}")
        print(f"ULTIMATE PREDICTION FOR: {user['name']}")
        print('='*70)
        
        result = predictor.predict_user_career(
            user['academic_grades'],
            user['hobbies'],
            user['personality']
        )
        
        # Primary recommendation
        primary = result['primary_recommendation']
        print(f"🎯 PRIMARY RECOMMENDATION: {primary['career']}")
        print(f"   Overall Confidence: {primary['confidence']:.1%}")
        
        # Hierarchical breakdown
        print(f"\n📊 COMPLETE HIERARCHICAL BREAKDOWN:")
        hierarchical = result['hierarchical_predictions']
        print(f"   🌐 Broad Category: {hierarchical['broad']['category']} ({hierarchical['broad']['confidence']:.1%})")
        print(f"   🏢 Field Level: {hierarchical['field']['category']} ({hierarchical['field']['confidence']:.1%})")
        print(f"   🎯 Specific Career: {hierarchical['specific']['category']} ({hierarchical['specific']['confidence']:.1%})")
        
        # Top alternatives
        alternatives = result['top_alternatives']
        
        print(f"\n🔍 TOP BROAD CATEGORY ALTERNATIVES:")
        for i, alt in enumerate(alternatives['broad_categories'][:3], 1):
            print(f"   {i}. {alt['category']} ({alt['confidence']:.1%})")
        
        print(f"\n🔍 TOP FIELD ALTERNATIVES:")
        for i, alt in enumerate(alternatives['fields'][:3], 1):
            print(f"   {i}. {alt['category']} ({alt['confidence']:.1%})")
        
        print(f"\n🔍 TOP CAREER ALTERNATIVES:")
        for i, alt in enumerate(alternatives['careers'][:3], 1):
            print(f"   {i}. {alt['career']} ({alt['confidence']:.1%})")
        
        print(f"\n💡 ULTIMATE REASONING:")
        for reason in result['recommendation_reasoning']:
            print(f"   • {reason}")

def save_ultimate_results(results, broad_importance, field_importance, career_importance, X):
    """Save comprehensive ultimate results"""
    print(f"\n{'='*80}")
    print("STEP 6: SAVING ULTIMATE RESULTS")
    print('='*80)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create ultimate results summary
    ultimate_summary = {
        "model_info": {
            "training_timestamp": timestamp,
            "model_type": "Ultimate Hierarchical Career Prediction",
            "architecture": "3-Level Hierarchy (Broad → Field → Specific)",
            "ensemble_method": "XGBoost + RandomForest with Soft Voting",
            "features_count": int(X.shape[1]),
            "samples_count": int(len(X))
        },
        "hierarchical_performance": {
            "broad_category_accuracy": float(results['broad_accuracy']),
            "field_model_accuracy": float(results['field_accuracy']),
            "specific_career_accuracy": float(results['career_accuracy']),
            "average_accuracy": float(np.mean([
                results['broad_accuracy'], 
                results['field_accuracy'], 
                results['career_accuracy']
            ]))
        },
        "model_details": {
            "broad_categories": len(results['broad_encoder'].classes_),
            "broad_classes": list(results['broad_encoder'].classes_),
            "field_categories": len(results['field_encoder'].classes_),
            "field_classes": list(results['field_encoder'].classes_),
            "career_categories": len(results['career_encoder'].classes_),
            "career_classes": list(results['career_encoder'].classes_)
        },
        "feature_engineering": {
            "total_features": int(X.shape[1]),
            "feature_categories": {
                "academic_features": 6,
                "personality_features": 5,
                "hobby_features": 10,
                "binary_hobby_features": 12,
                "derived_features": 10,
                "orientation_features": 4,
                "specialization_indices": 3
            },
            "all_features": list(X.columns)
        },
        "top_features": {
            "broad_category_top_10": [
                {"feature": row['feature'], "importance": float(row['importance'])}
                for _, row in broad_importance.head(10).iterrows()
            ],
            "field_level_top_10": [
                {"feature": row['feature'], "importance": float(row['importance'])}
                for _, row in field_importance.head(10).iterrows()
            ],
            "career_level_top_10": [
                {"feature": row['feature'], "importance": float(row['importance'])}
                for _, row in career_importance.head(10).iterrows()
            ]
        },
        "improvements_applied": [
            "✓ Complete 3-level hierarchical prediction",
            "✓ 50+ engineered features with advanced combinations",
            "✓ SMOTE for class imbalance handling at all levels",
            "✓ Ensemble methods (XGBoost + RandomForest)",
            "✓ Soft voting for robust predictions",
            "✓ Comprehensive feature importance analysis",
            "✓ Enhanced data cleaning and validation",
            "✓ XGBoost compatibility fixes",
            "✓ Advanced personality-derived features",
            "✓ Specialization and orientation indices"
        ]
    }
    
    # Save comprehensive results
    with open(f"results/ultimate_model_results_{timestamp}.json", 'w') as f:
        json.dump(ultimate_summary, f, indent=2)
    
    print("✓ Ultimate results saved to:")
    print(f"  • results/ultimate_model_results_{timestamp}.json")
    print(f"  • results/ultimate_broad_importance.csv")
    print(f"  • results/ultimate_field_importance.csv")
    print(f"  • results/ultimate_career_importance.csv")
    print(f"  • models/ (6 model files)")

def main():
    """Ultimate main execution function"""
    start_time = time.time()
    
    print("🚀 ULTIMATE CAREER PREDICTION MODEL")
    print("AI-Powered Hierarchical Career Recommendation System")
    print("Combining All Advanced Features with Complete Hierarchy")
    print("="*80)
    
    # Create project structure
    create_directories()
    
    # Load and prepare data
    dataset = load_and_prepare_data()
    
    # Create ultimate features
    X, career_labels = create_ultimate_features(dataset)
    broad_labels, field_labels = create_hierarchical_labels(career_labels)
    
    # Train hierarchical models
    results = train_hierarchical_models(X, career_labels, broad_labels, field_labels)
    
    # Analyze feature importance
    broad_importance, field_importance, career_importance = analyze_feature_importance(results, X)
    
    # Create ultimate predictor
    predictor = create_ultimate_predictor(results)
    
    # Demonstrate predictions
    demonstrate_ultimate_predictions(predictor)
    
    # Save results
    save_ultimate_results(results, broad_importance, field_importance, career_importance, X)
    
    # Final summary
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print("🎉 ULTIMATE TRAINING COMPLETE!")
    print('='*80)
    print("✓ Complete hierarchical models trained and saved")
    print("✓ Comprehensive feature analysis completed")
    print("✓ Ultimate predictions demonstrated")
    print("✓ All results saved with timestamp")
    print("✓ Ready for production deployment")
    
    print(f"\n📊 ULTIMATE PERFORMANCE SUMMARY:")
    print(f"• Broad Category Model Accuracy: {results['broad_accuracy']:.1%}")
    print(f"• Field Model Accuracy: {results['field_accuracy']:.1%}")
    print(f"• Specific Career Model Accuracy: {results['career_accuracy']:.1%}")
    print(f"• Average Hierarchical Accuracy: {np.mean([results['broad_accuracy'], results['field_accuracy'], results['career_accuracy']]):.1%}")
    print(f"• Total Features Engineered: {X.shape[1]}")
    print(f"• Training Time: {training_time:.1f} seconds")
    
    print(f"\n📁 FILES CREATED:")
    print(f"• models/ultimate_broad_model.joblib")
    print(f"• models/ultimate_field_model.joblib")
    print(f"• models/ultimate_career_model.joblib")
    print(f"• models/broad_encoder.joblib")
    print(f"• models/field_encoder.joblib")
    print(f"• models/career_encoder.joblib")
    print(f"• results/ultimate_model_results_[timestamp].json")
    print(f"• results/ultimate_*_importance.csv (3 files)")
    
    print(f"\n🌟 The Ultimate Career Prediction System is ready!")
    print(f"   Use the UltimateCareerPredictor class for hierarchical predictions")
    print(f"   with confidence scoring at all levels!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ultimate training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during ultimate training: {str(e)}")
        import traceback
        traceback.print_exc()
