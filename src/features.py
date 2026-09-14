"""
Stage 3 — Feature engineering.

Converts raw candidate/job fields into numeric features a model can learn from.
CRITICAL: skill NAMES never become features (a job asking for Python and a job
asking for Arduino must produce comparable feature vectors) -- only relative
measures do: match ratio, listed count, focus ratio, distances.
"""

import pandas as pd
from data_process import extract_row_features

EDU_LEVELS = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
TARGET_EXPERIENCE = 1.5
TARGET_EDU_VAL = EDU_LEVELS['Master']

FEATURE_COLUMNS = [
    'experience_distance',
    'education_distance',
    'skill_match_ratio',
    'skill_num_listed',
    'skill_focus_ratio',
    'cover_letter_similarity',
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    feats['experience_distance'] = (df['experience_years'] - TARGET_EXPERIENCE).abs()

    edu_val = df['education_level'].map(EDU_LEVELS).fillna(0)
    feats['education_distance'] = (edu_val - TARGET_EDU_VAL).abs()

    feats['skill_match_ratio'] = df['num_matched'] / df['num_requested'].clip(lower=1)
    feats['skill_num_listed'] = df['num_listed']
    feats['skill_focus_ratio'] = df['num_matched'] / df['num_listed'].clip(lower=1)

    feats['cover_letter_similarity'] = df['cover_letter_similarity']

    return feats[FEATURE_COLUMNS]

def build_features_from_raw(df: pd.DataFrame)-> pd.DataFrame:
    raw_feats = df.apply(extract_row_features, axis=1, result_type='expand')
    return raw_feats[FEATURE_COLUMNS]