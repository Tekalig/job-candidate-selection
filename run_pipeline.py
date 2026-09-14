# --- modify run_pipeline.py ---

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_synthetic_dataset
from train import train_and_evaluate
from predict import rank_candidates, sample_new_candidates

# Set this once Elias delivers his file
USE_ELIAS_DATA = True
ELIAS_CSV_PATH = 'data/synthetic_candidates.csv'   # his raw-text CSV goes here


def main():
    if USE_ELIAS_DATA:
        print("Using Elias's raw-text dataset:", ELIAS_CSV_PATH)
        csv_path = ELIAS_CSV_PATH
        data_format = 'raw'
    else:
        print("Generating synthetic numeric dataset")
        df = generate_synthetic_dataset(n_random=480, seed=42)
        os.makedirs('data', exist_ok=True)
        csv_path = 'data/synthetic_candidates.csv'
        df.to_csv(csv_path, index=False)
        data_format = 'synthetic'

    os.makedirs('models', exist_ok=True)
    results = train_and_evaluate(csv_path=csv_path, model_out_path='models/fit_model.joblib', data_format=data_format)
    print(f"Test MAE: {results['mae']:.4f}   Test R^2: {results['r2']:.4f}")
    print(results['feature_importances'].to_string())

    new_df = sample_new_candidates()  # note: needs raw-text fields if data_format='raw', see below
    ranked = rank_candidates('models/fit_model.joblib', new_df, data_format=data_format)
    print(ranked.to_string(index=False))


if __name__ == '__main__':
    main()