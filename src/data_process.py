import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Reused from ideal_profile.py — same formulas, same weights
# ============================================================

EDU_LEVELS = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
TARGET_EDU_LEVEL = 'Master'
TARGET_EXPERIENCE_YEARS = 1.5

WEIGHTS = {'experience': 0.35, 'education': 0.15, 'skills': 0.35, 'motivation': 0.15}


def score_experience(exp_years, target=TARGET_EXPERIENCE_YEARS, tolerance=1.3):
    return float(np.exp(-0.5 * ((exp_years - target) / tolerance) ** 2))


def score_education(edu_level, target=TARGET_EDU_LEVEL):
    cand_val = EDU_LEVELS.get(edu_level, None)
    target_val = EDU_LEVELS.get(target, 2)
    if cand_val is None:
        return 0.0
    distance = abs(cand_val - target_val)
    return max(0.0, 1.0 - 0.3 * distance)


def score_skills(matched, num_requested, num_listed):
    if num_requested <= 0:
        return 0.0
    match_ratio = matched / num_requested
    extra_skills = max(0, num_listed - matched)
    focus_bonus = 1.0 / (1.0 + 0.15 * extra_skills)
    return float(match_ratio * focus_bonus)


def score_motivation(cover_letter_similarity, strong_threshold=0.75):
    sim = max(0.0, min(1.0, cover_letter_similarity))
    return sim if sim >= strong_threshold else sim * 0.85


# ============================================================
# NEW — parsing raw skill strings + real text similarity
# ============================================================

def parse_skills(skill_string: str) -> list:
    """'Python,SQL,Docker' -> ['Python', 'SQL', 'Docker']"""
    if pd.isna(skill_string) or not str(skill_string).strip():
        return []
    return [s.strip() for s in str(skill_string).split(',') if s.strip()]


def compute_text_similarity(job_description, cover_letter_text):
    job_description = "" if pd.isna(job_description) else str(job_description)
    cover_letter_text = "" if pd.isna(cover_letter_text) else str(cover_letter_text)

    if not job_description.strip() or not cover_letter_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf = vectorizer.fit_transform([
        job_description,
        cover_letter_text
    ])

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return float(similarity)


# ============================================================
# Row-level feature extraction + label computation
# ============================================================

def extract_row_features(row: pd.Series) -> dict:
    """
    Turns one raw data row (Elias's format: candidate_skills, required_skills,
    cover_letter_text, job_description) into the numeric features the model needs.
    """
    candidate_skills = parse_skills(row.get('candidate_skills', row.get('skills', '')))
    required_skills = parse_skills(row.get('required_skills', ''))

    matched = len(set(candidate_skills).intersection(set(required_skills)))
    num_requested = len(required_skills)
    num_listed = len(candidate_skills)

    cover_sim = compute_text_similarity(
        row.get('cover_letter_text', ''), row.get('job_description', '')
    )

    exp_val = row['experience_years']
    edu_val = EDU_LEVELS.get(row['education_level'], 0)

    return {
        'experience_distance': abs(exp_val - TARGET_EXPERIENCE_YEARS),
        'education_distance': abs(edu_val - EDU_LEVELS[TARGET_EDU_LEVEL]),
        'skill_match_ratio': matched / max(num_requested, 1),
        'skill_num_listed': num_listed,
        'skill_focus_ratio': matched / max(num_listed, 1),
        'cover_letter_similarity': cover_sim,
        # kept around for label computation / debugging, not fed to the model directly
        '_matched': matched,
        '_num_requested': num_requested,
        '_num_listed': num_listed,
    }


def compute_fit_label(row: pd.Series, weights: dict = WEIGHTS) -> dict:
    """
    Computes the fit_label from raw row data, using the SAME formulas as
    ideal_profile.py. Use this to (re)generate labels for Elias's dataset
    consistently, or to sanity-check a fit_label he already included.
    """
    feats = extract_row_features(row)

    sub_scores = {
        'experience': score_experience(row['experience_years']),
        'education': score_education(row['education_level']),
        'skills': score_skills(feats['_matched'], feats['_num_requested'], feats['_num_listed']),
        'motivation': score_motivation(feats['cover_letter_similarity']),
    }
    final = sum(sub_scores[k] * weights[k] for k in weights)
    return {'computed_fit_label': round(final, 4), 'sub_scores': sub_scores,
            'cover_letter_similarity': feats['cover_letter_similarity']}


# ============================================================
# Apply to a full dataframe
# ============================================================

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Adds feature columns + a recomputed fit label to every row."""
    feature_rows = df.apply(extract_row_features, axis=1, result_type='expand')
    label_rows = df.apply(lambda r: pd.Series(compute_fit_label(r)['sub_scores']), axis=1)
    label_rows['computed_fit_label'] = df.apply(lambda r: compute_fit_label(r)['computed_fit_label'], axis=1)

    result = pd.concat([df, feature_rows, label_rows], axis=1)
    return result


# ============================================================
# Example run on Elias's sample row
# ============================================================

if __name__ == '__main__':
    sample_data = {
        'candidate_id': ['C0001'],
        'archetype': ['perfect'],
        'experience_years': [1.5],
        'education_level': ['Master'],
        'candidate_skills': ['Python,SQL,Docker'],
        'required_skills': ['Python,SQL,Docker'],   # <- Elias's new column
        'cover_letter_text': ["Dear Hiring Team, I am excited to apply for the Data Engineer position. "
                               "With 1.5 years of hands-on experience and a Master's degree, I have built "
                               "production data pipelines using Python, SQL, and Docker."],
        'job_description': ["We are seeking a Data Engineer with 1.5 years of experience and a Master's "
                             "degree. Required skills: Python, SQL, Docker."],
        'fit_label': [0.9654],  # Elias's provided label
    }
    df = pd.DataFrame(sample_data)

    processed = process_dataset(df)
    print(processed.to_string(index=False,))