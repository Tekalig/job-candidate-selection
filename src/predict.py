"""
Stage 4b — Score and rank new, unseen candidates using the trained model.
"""

import joblib
import pandas as pd

from features import build_features


def rank_candidates(model_path: str, new_candidates_df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(model_path)
    X_new = build_features(new_candidates_df)
    new_candidates_df = new_candidates_df.copy()
    new_candidates_df['predicted_fit'] = model.predict(X_new).round(4)
    return new_candidates_df.sort_values('predicted_fit', ascending=False)


def sample_new_candidates() -> pd.DataFrame:
    """A handful of new candidates the model has never seen, across different jobs."""
    return pd.DataFrame([
        {
            'name': 'Near-ideal, focused (Python job)',
            'experience_years': 1.6, 'education_level': 'Master',
            'num_matched': 3, 'num_requested': 3, 'num_listed': 3,
            'cover_letter_similarity': 0.88,
        },
        {
            'name': 'Senior, broad skills, generic letter (Python job)',
            'experience_years': 8.0, 'education_level': 'PhD',
            'num_matched': 3, 'num_requested': 3, 'num_listed': 9,
            'cover_letter_similarity': 0.35,
        },
        {
            'name': 'Junior, under-skilled but motivated (Embedded job)',
            'experience_years': 0.4, 'education_level': 'Bachelor',
            'num_matched': 1, 'num_requested': 3, 'num_listed': 1,
            'cover_letter_similarity': 0.91,
        },
        {
            'name': 'Right experience, wrong education (Backend job)',
            'experience_years': 1.4, 'education_level': 'Bachelor',
            'num_matched': 3, 'num_requested': 3, 'num_listed': 4,
            'cover_letter_similarity': 0.79,
        },
        {
            'name': 'Perfect on paper, borderline cover letter (ML job)',
            'experience_years': 1.5, 'education_level': 'Master',
            'num_matched': 3, 'num_requested': 3, 'num_listed': 3,
            'cover_letter_similarity': 0.60,
        },
    ])


if __name__ == '__main__':
    new_df = sample_new_candidates()
    ranked = rank_candidates('../models/fit_model.joblib', new_df)
    print(ranked[['name', 'predicted_fit']].to_string(index=False))
