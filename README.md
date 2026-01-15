# Fashion Recommendation System

A recommendation system for fashion e-commerce.

## Setup
1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
3. Install dependencies: `pip install -r requirements.txt`
4. Configure `.env` with database credentials
5. Run scripts in order: 02, 03, 04, 05

## Project Structure
- `config/` - Configuration files
- `data/` - Data storage (raw, processed, exports)
- `models/` - Trained models
- `src/` - Source code
- `scripts/` - Execution scripts
- `notebooks/` - Jupyter notebooks
- `tests/` - Unit tests
- `logs/` - Log files

## Usage
```bash
# Load data from PostgreSQL
python scripts/02_load_data.py

# Preprocess data
python scripts/03_preprocess_data.py

# Train models
python scripts/04_train_models.py

# Generate recommendations
python scripts/05_generate_recommendations.py
```
