"""
End-to-end pipeline runner.
Stage 1+2: generate labeled synthetic data
Stage 3+4: engineer features, train, evaluate
Stage 5:   score + rank new, unseen candidates
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_synthetic_dataset
from train import train_and_evaluate
from predict import rank_candidates, sample_new_candidates


def main():
    print("=" * 60)
    print("STAGE 1+2: Generating synthetic labeled dataset")
    print("=" * 60)
    df = generate_synthetic_dataset(n_random=480, seed=42)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_candidates.csv', index=False)
    print(f"Generated {len(df)} candidates -> data/synthetic_candidates.csv")
    print(f"Label range: {df['fit_label'].min():.3f} to {df['fit_label'].max():.3f}")
    print(f"Label mean:  {df['fit_label'].mean():.3f}\n")

    print("=" * 60)
    print("STAGE 3+4: Training supervised model")
    print("=" * 60)
    os.makedirs('models', exist_ok=True)
    results = train_and_evaluate(
        csv_path='data/synthetic_candidates.csv',
        model_out_path='models/fit_model.joblib'
    )
    print(f"Train size: {results['n_train']}   Test size: {results['n_test']}")
    print(f"Test MAE:   {results['mae']:.4f}  (avg prediction error, lower is better)")
    print(f"Test R^2:   {results['r2']:.4f}  (variance explained, closer to 1 is better)")
    print("\nFeature importances (what the model actually learned to weigh):")
    print(results['feature_importances'].to_string())
    print()

    print("=" * 60)
    print("STAGE 5: Scoring new, unseen candidates")
    print("=" * 60)
    new_df = sample_new_candidates()
    ranked = rank_candidates('models/fit_model.joblib', new_df)
    print(ranked[['name', 'experience_years', 'education_level',
                   'num_matched', 'num_requested', 'num_listed',
                   'cover_letter_similarity', 'predicted_fit']].to_string(index=False))

    ranked.to_csv('outputs/new_candidate_predictions.csv', index=False)
    print("\nSaved -> outputs/new_candidate_predictions.csv")


if __name__ == '__main__':
    main()
