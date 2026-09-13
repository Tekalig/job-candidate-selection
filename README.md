# Candidate Fit Model — End-to-End Pipeline

Supervised ML model that learns Ben's "ideal candidate" preferences from
synthetic labeled data, then scores and ranks new candidates.

## Structure
```
candidate_fit_model/
├── src/
│   ├── ideal_profile.py   # Stage 1: labeling rules (used only to generate training labels)
│   ├── data_generator.py  # Stage 2: synthetic candidate/job data generator
│   ├── features.py        # Stage 3: job-agnostic feature engineering
│   ├── train.py            # Stage 4: train + evaluate supervised model
│   └── predict.py          # Stage 5: score + rank new candidates
├── data/synthetic_candidates.csv     # generated training data
├── models/fit_model.joblib           # trained model
├── outputs/new_candidate_predictions.csv
├── run_pipeline.py         # runs everything end to end
└── requirements.txt
```

## Run it
```
pip install -r requirements.txt
python run_pipeline.py
```

## Key design point
`ideal_profile.py`'s rules are called ONLY inside `data_generator.py`, to
LABEL synthetic data. The model in `train.py` never sees those rules
directly — it only sees engineered features (features.py) and the label,
and learns the mapping itself. This is what makes it a real supervised
model rather than the original rule-based scorer with a new name.

## Skills are job-agnostic by design
Skill NAMES are never features — only match_ratio, listed count, and
focus_ratio. This lets one model serve jobs requesting completely
different skill sets (Python vs. Arduino vs. FastAPI) without retraining.
