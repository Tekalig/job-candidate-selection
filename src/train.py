"""
Stage 4a — Train and evaluate the supervised fit-score model.

Splits the labeled synthetic dataset into train/test, trains a real
regression model (never sees ideal_profile.py's rules directly -- only
the engineered features and the label), and reports test performance
plus feature importances to check whether it learned the intended pattern.
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from features import build_features, FEATURE_COLUMNS


def train_and_evaluate(csv_path: str, model_out_path: str):
    df = pd.read_csv(csv_path)

    X = build_features(df)
    y = df['fit_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)

    joblib.dump(model, model_out_path)

    return {
        'mae': mae,
        'r2': r2,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_importances': importances,
    }


if __name__ == '__main__':
    results = train_and_evaluate(
        csv_path='../data/synthetic_candidates.csv',
        model_out_path='../models/fit_model.joblib'
    )
    print(f"Train size: {results['n_train']}  Test size: {results['n_test']}")
    print(f"Test MAE:  {results['mae']:.4f}")
    print(f"Test R^2:  {results['r2']:.4f}")
    print("\nFeature importances:")
    print(results['feature_importances'].to_string())
