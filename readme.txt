# 🏏 IPL Win Probability Predictor

A machine learning app that predicts the probability of a team winning an IPL match in real time, based on live match state and historical team/venue data.

🔗 **Live demo:** [ipl-win-predictor-by-dshnn.streamlit.app](https://ipl-win-predictor-by-dshnn.streamlit.app/)  
💻 **Code:** [github.com/dshnn](https://github.com/dshnn)

---

## What it does

Select two teams, a venue, and the current match state (over, runs, wickets, target) — the app returns a live win probability for team 1, a visual gauge, and a win probability curve showing how the prediction evolved over the course of the innings.

Historical features (head-to-head record, team win rates, venue-specific win rates, average first-innings scores) are auto-filled from the dataset — the user never has to enter them manually.

---

## How it works

### Data pipeline

The project uses two datasets: `matches.csv` (one row per match) and `deliveries.csv` (one row per ball bowled). These were combined into a single modeling table through a multi-step pipeline:

1. **Standardised team names** across all 16 seasons (e.g. Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings) to preserve franchise continuity in historical features.
2. **Sorted and compressed** `deliveries.csv` to one row per over per innings using `groupby(['match_id', 'inning', 'over']).last()` — reducing ~200 rows per match to ~20.
3. **Computed cumulative match-state features** per ball: `cum_runs`, `cum_wickets`, `balls`, and derived features including `current_run_rate`, `required_run_rate`, `wicket_in_hands`, and `balls_remaining`.
4. **Merged match context** from `matches.csv` onto the compressed delivery table using `match_id` as the join key.
5. **Computed historical features chronologically** — using a sequential loop over matches sorted by date, maintaining running dictionaries for team win rates, venue win rates, head-to-head records, and average first-innings scores per venue. For each match, features were read *before* the match result was added to the dictionaries, preventing any leakage of future match outcomes into historical features.

### Feature set

| Category | Features |
|---|---|
| Match state | `cum_runs`, `cum_wickets`, `balls`, `balls_remaining`, `current_run_rate`, `required_run_rate`, `wicket_in_hands`, `req_runs` |
| Match context | `inning`, `over`, `target_runs`, `target_overs` |
| Toss | `t1_won_toss` |
| Historical | `h2h`, `team1_win_rate`, `team2_win_rate`, `team1_wr_at_venue`, `team2_wr_at_venue`, `avg_score_of_stadium` |
| Encoded | Team 1 (OHE), Team 2 (OHE), Venue (OHE) — 80+ binary columns |

### Target variable

`team1_won` — binary (1 if `team1` won the match, else 0). Consistent across all rows of the same match.

### Train / test split

Split **chronologically by season** — earlier seasons for training, most recent 1–2 seasons for testing. A random split was deliberately avoided because rows from the same match would otherwise appear in both train and test sets, leaking information about the match outcome across the split boundary.

### Model

**XGBoost Classifier** — chosen over Logistic Regression and Random Forest due to its ability to model non-linear interactions between features (e.g. the interaction between `balls_remaining` and `wicket_in_hands` is more predictive than either feature alone, something tree-based models capture naturally).

---

## Model performance

> ⚠️ **Known issue:** Class 1 recall is low (0.21) due to class imbalance in the dataset (~78% class 0, ~22% class 1). The model currently predicts the majority class more readily. This will be addressed in the next version using `scale_pos_weight` in XGBoost to upweight the minority class during training. The AUC of 0.849 confirms the model has genuine discriminative ability — the recall issue is a threshold/imbalance problem, not a fundamental model quality problem.

| Metric | Class 0 (team 1 loses) | Class 1 (team 1 wins) | Weighted avg |
|---|---|---|---|
| Precision | 0.81 | 0.62 | 0.77 |
| Recall | 0.96 | 0.21 | 0.80 |
| F1-score | 0.88 | 0.32 | 0.76 |
| **AUC** | — | — | **0.849** |

*Evaluated on a chronological held-out test set (most recent seasons).*

---

## Key engineering decisions and findings

**Chronological loop for historical features**  
A naive `groupby` to compute team win rates and head-to-head records would use the entire dataset — including matches that happen *after* the match being predicted. The fix: process matches one by one in date order, reading the running statistics before each match and updating them only after. This ensures every historical feature value reflects only information available at that point in time.

**One row per over, not per ball**  
Rather than keeping all ~200 rows per match, the dataset was compressed to ~20 rows (one per completed over). This gives the model the right prediction granularity — "win probability after over 12" — without the noise of individual ball-level variation, and reduces dataset size by ~6x.

**Inning-aware features**  
`required_run_rate` is only meaningful in the second innings. In the first innings it defaults to a sentinel value. The model uses `inning` as a feature to contextualise whether `required_run_rate` is informative, allowing it to learn inning-specific patterns independently.

**Team name standardisation**  
Without standardising franchise names across seasons, the model would treat "Delhi Daredevils" and "Delhi Capitals" as entirely different teams — losing 6+ years of that franchise's historical win rate and venue data. Standardisation was applied before any feature computation.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, numpy |
| Modelling | scikit-learn, XGBoost |
| Visualisation | Plotly |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |
| Model persistence | joblib, pickle |

---

## Project structure

```
ipl-win-predictor/
├── app.py              # Streamlit frontend
├── ipl_model.pkl       # Trained XGBoost model
├── lookup.pkl          # Pre-computed historical feature lookup tables
├── requirements.txt    # Dependencies
├── notebook.ipynb      # Full data pipeline and model training
└── README.md
```

---

## Run locally

```bash
git clone https://github.com/dshnn/ipl-win-predictor
cd ipl-win-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

IPL Complete Dataset (2008–2023) — sourced from Kaggle.  
Contains `matches.csv` and `deliveries.csv` covering all IPL seasons.

---

## What's next

- [ ] Fix class imbalance using `scale_pos_weight` in XGBoost and re-evaluate
- [ ] Lower prediction threshold from 0.5 to 0.3–0.35 to improve minority class recall
- [ ] Add win probability curve for a full historical match replay
- [ ] Add player-level features (top batter/bowler current form)
- [ ] Add recent form feature (team's win rate in last 5 matches)

---

## Author

Built by [@dshnn](https://github.com/dshnn) as part of a self-driven AI/ML portfolio.  
Project 2 of 3 in a structured portfolio build — from raw data to deployed product.