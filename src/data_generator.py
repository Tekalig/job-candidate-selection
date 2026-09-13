"""
Stage 2 — Synthetic data generation.

Generates candidate/job pairs with realistic variation across all four
dimensions, then labels each one using ideal_profile.ideal_profile_score().
The skill pool varies PER JOB, so the model never sees a fixed skill
vocabulary -- it has to learn the pattern of "focused relevance",
not that any specific skill is good.
"""

import numpy as np
import pandas as pd

from ideal_profile import ideal_profile_score, EDU_LEVELS

# Several distinct skill pools, simulating different job families.
SKILL_POOLS = [
    ['Python', 'PyTorch', 'SQL', 'Pandas', 'Docker', 'AWS'],
    ['Java', 'Arduino', 'C++', 'Embedded C', 'RTOS', 'CAN Bus'],
    ['FastAPI', 'PostgreSQL', 'Docker', 'Redis', 'Kubernetes', 'gRPC'],
    ['PyTorch', 'Transformers', 'RAG', 'LangChain', 'Vector DB', 'CUDA'],
    ['React', 'TypeScript', 'Node.js', 'GraphQL', 'Next.js', 'Tailwind'],
    ['Excel', 'SQL', 'Tableau', 'PowerBI', 'Statistics', 'A/B Testing'],
]

EDU_CHOICES = list(EDU_LEVELS.keys())
EDU_PROBS = [0.35, 0.45, 0.20]  # Bachelor, Master, PhD


def _sample_job_spec(rng):
    pool = pool = SKILL_POOLS[rng.integers(0, len(SKILL_POOLS))]
    num_requested = rng.integers(2, 5)  # jobs request 2-4 skills
    requested = list(rng.choice(pool, size=num_requested, replace=False))
    return {'pool': pool, 'requested_skills': requested}


def _sample_candidate(rng, job, edge_case=None):
    pool = job['pool']
    requested = job['requested_skills']
    num_requested = len(requested)

    if edge_case == 'perfect':
        exp = 1.5
        edu = 'Master'
        matched_skills = list(requested)
        extra = []
        cover_sim = float(rng.uniform(0.85, 0.98))
    elif edge_case == 'stuffer':
        exp = float(rng.normal(1.5, 0.3))
        edu = 'Master'
        matched_skills = list(requested)
        non_req = [s for s in pool if s not in requested]
        extra = list(rng.choice(non_req, size=min(len(non_req), rng.integers(3, 6)), replace=False))
        cover_sim = float(rng.uniform(0.2, 0.45))  # generic letter
    elif edge_case == 'underqualified':
        exp = float(rng.choice([0.2, 0.3, 5.0, 7.0]))  # far from 1.5, either direction
        edu = rng.choice(['Bachelor', 'PhD'])
        num_matched = rng.integers(0, max(1, num_requested - 1))
        matched_skills = list(rng.choice(requested, size=num_matched, replace=False)) if num_matched > 0 else []
        extra = []
        cover_sim = float(rng.uniform(0.3, 0.6))
    else:
        # realistic random candidate
        exp = max(0.0, float(rng.normal(1.5, 2.5)))
        edu = rng.choice(EDU_CHOICES, p=EDU_PROBS)
        num_matched = rng.integers(0, num_requested + 1)
        matched_skills = list(rng.choice(requested, size=num_matched, replace=False)) if num_matched > 0 else []
        non_req = [s for s in pool if s not in requested]
        extra_count = rng.integers(0, 4)
        extra = list(rng.choice(non_req, size=min(extra_count, len(non_req)), replace=False)) if len(non_req) > 0 else []
        cover_sim = float(np.clip(rng.beta(2, 2), 0, 1))

    candidate_skills = list(matched_skills) + list(extra)
    matched = len(set(candidate_skills).intersection(set(requested)))

    return {
        'experience_years': round(float(exp), 2),
        'education_level': edu,
        'requested_skills': ','.join(requested),
        'candidate_skills': ','.join(candidate_skills),
        'num_matched': matched,
        'num_requested': num_requested,
        'num_listed': len(candidate_skills),
        'cover_letter_similarity': round(cover_sim, 3),
    }


def generate_synthetic_dataset(n_random: int = 480, seed: int = 42) -> pd.DataFrame:
    """
    Generates n_random realistic candidates plus a fixed set of hand-crafted
    edge cases (perfect match, keyword-stuffer, under-qualified), all labeled
    via the SAME formula in ideal_profile.py for consistency across the dataset.
    """
    rng = np.random.default_rng(seed)
    rows = []

    edge_case_types = ['perfect', 'stuffer', 'underqualified'] * 7  # 21 hand-crafted cases
    for ec in edge_case_types:
        job = _sample_job_spec(rng)
        cand = _sample_candidate(rng, job, edge_case=ec)
        rows.append(cand)

    for _ in range(n_random):
        job = _sample_job_spec(rng)
        cand = _sample_candidate(rng, job, edge_case=None)
        rows.append(cand)

    df = pd.DataFrame(rows)

    labels = []
    for _, row in df.iterrows():
        result = ideal_profile_score(
            exp_years=row['experience_years'],
            edu_level=row['education_level'],
            matched=row['num_matched'],
            num_requested=row['num_requested'],
            num_listed=row['num_listed'],
            cover_letter_similarity=row['cover_letter_similarity'],
        )
        labels.append(result['fit_label'])

    df['fit_label'] = labels
    df.insert(0, 'candidate_id', [f'C{i+1:03d}' for i in range(len(df))])
    return df


if __name__ == '__main__':
    df = generate_synthetic_dataset()
    df.to_csv('../data/synthetic_candidates.csv', index=False)
    print(f"Generated {len(df)} synthetic candidates")
    print(df.head(10).to_string())
    print("\nLabel distribution:")
    print(df['fit_label'].describe())
