"""
Stage 1 — Ideal profile logic.

This module encodes Ben's stated preferences into scoring functions.
IMPORTANT: this logic is used ONLY to generate labels for synthetic
training data (see data_generator.py). The trained model in train.py
never calls these functions directly at prediction time — it learns
the mapping from features to label on its own.

Ben's stated preferences:
- Experience: 1.5 years is the best fit. More or less is NOT better.
- Education: Master is the best fit. Bachelor or PhD lose some score.
- Skills: required skills vary by job. Reward matching requested skills;
  reward focus (fewer total listed skills relative to matches) over breadth.
- Motivation: cover letter should strongly match the target job
  description. ~0.75+ cosine similarity is treated as strong,
  job-specific motivation.
"""

import numpy as np

EDU_LEVELS = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
TARGET_EDU_LEVEL = 'Master'
TARGET_EXPERIENCE_YEARS = 1.5

# Weights must sum to 1.0
WEIGHTS = {
    'experience': 0.35,
    'education': 0.15,
    'skills': 0.35,
    'motivation': 0.15,
}


def score_experience(exp_years: float, target: float = TARGET_EXPERIENCE_YEARS, tolerance: float = 1.3) -> float:
    """Gaussian bell curve peaking at `target`. Symmetric penalty for deviation
    in either direction — more experience is NOT automatically better."""
    return float(np.exp(-0.5 * ((exp_years - target) / tolerance) ** 2))


def score_education(edu_level: str, target: str = TARGET_EDU_LEVEL) -> float:
    """Peak at target education level. Distance-based penalty, symmetric —
    both under- and over-shooting the target lose score."""
    cand_val = EDU_LEVELS.get(edu_level, None)
    target_val = EDU_LEVELS.get(target, 2)
    if cand_val is None:
        return 0.0  # unrecognized education -> flagged as invalid, not silently guessed
    distance = abs(cand_val - target_val)
    return max(0.0, 1.0 - 0.3 * distance)


def score_skills(matched: int, num_requested: int, num_listed: int) -> float:
    """
    Skill fit = f(match ratio, focus).
    Requested skill NAMES never enter this formula — only counts —
    so the same function works regardless of which skills a job requests.

    match_ratio: how much of what was asked for is covered.
    focus_bonus: penalizes listing many extra skills beyond what's requested
    (broad "spray and pray" skill lists dilute apparent core strength).
    """
    if num_requested <= 0:
        return 0.0
    match_ratio = matched / num_requested
    extra_skills = max(0, num_listed - matched)
    focus_bonus = 1.0 / (1.0 + 0.15 * extra_skills)
    return float(match_ratio * focus_bonus)


def score_motivation(cover_letter_similarity: float, strong_threshold: float = 0.75) -> float:
    """
    Cosine similarity between cover letter and job description.
    Values at/above strong_threshold are treated as genuine, job-specific
    motivation. Below threshold gets a mild (not cliff-edge) discount.
    """
    sim = max(0.0, min(1.0, cover_letter_similarity))
    if sim >= strong_threshold:
        return sim
    return sim * 0.85  # smooth discount, no hard cliff


def ideal_profile_score(exp_years, edu_level, matched, num_requested, num_listed,
                         cover_letter_similarity, weights: dict = None) -> dict:
    """Combine all four dimensions into sub-scores + final weighted label."""
    w = weights or WEIGHTS
    assert abs(sum(w.values()) - 1.0) < 1e-6, "weights must sum to 1.0"

    sub_scores = {
        'experience': score_experience(exp_years),
        'education': score_education(edu_level),
        'skills': score_skills(matched, num_requested, num_listed),
        'motivation': score_motivation(cover_letter_similarity),
    }
    final = sum(sub_scores[k] * w[k] for k in w)
    return {'fit_label': round(final, 4), 'sub_scores': sub_scores}
