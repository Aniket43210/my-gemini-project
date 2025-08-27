# AGENTS.md - Career Prediction System

## Build/Test/Lint Commands
- **Train main model**: `python main.py`
- **Run all tests**: `python run_tests.py`
- **Run specific test**: `python run_tests.py test_data_generator`
- **Debug tests**: `python debug_tests.py`
- **Build for deployment**: `bash build.sh` (installs dependencies)
- **Install dependencies**: `pip install -r requirements.txt`
- **Run web app**: `python app.py` (Flask server on http://127.0.0.1:5000)
- **Test predictor**: `python simple_predictor.py`

## Architecture & Codebase Structure
- **Python ML project** with hierarchical career prediction using XGBoost + RandomForest
- **Main training script**: `main.py` - 3-level hierarchy (Broad→Field→Specific careers)
- **Source modules**: `src/` - data generation, feature engineering, ensemble models, hyperparameter tuning
- **Test suite**: `tests/` - comprehensive unit tests for all components
- **Models directory**: `models/` - stores trained models (.joblib files)
- **Data directory**: `data/` - JSON training datasets (synthetic_career_data.json, enhanced_career_data.json)
- **Results directory**: `results/` - model performance analysis and feature importance

## Code Style & Conventions
- **Import style**: Standard library first, third-party, then local modules with `sys.path.append('src')`
- **Functions**: Snake_case with comprehensive docstrings
- **Classes**: PascalCase (e.g., `SyntheticDataGenerator`, `AdvancedHyperparameterTuner`)
- **Error handling**: Try-catch with graceful fallbacks, especially for data cleaning
- **Data validation**: `safe_float()` function for robust numeric conversion with defaults
- **Feature naming**: Descriptive names like `stem_vs_humanities`, `leadership_potential`
- **File structure**: Organized into logical directories with clear separation of concerns

## Important Notes
- **Flask web app**: `app.py` provides web interface for career prediction with HTML forms
- **Demo mode**: App falls back to rule-based predictions if ML models fail to load
- **Test framework**: Uses standard Python `unittest` module  
- **Data format**: JSON files with academic_grades, hobbies, personality, career fields
- **Dependencies**: Heavy ML stack (pandas, scikit-learn, xgboost, matplotlib, flask)
- **Templates**: HTML templates in `templates/` directory with Bootstrap UI
