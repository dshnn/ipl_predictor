import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="wide"
)

# ── load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("ipl_model.pkl")

model = load_model()
st.sidebar.success(f"Loaded model with {len(model.feature_names_in_)} features")

# ── load lookup tables from your dataset ────────────────────────────────────
# Replace this with your actual pre-computed values from the dataset
# Run this ONCE in your notebook and save it:
#
#   import pickle
#   lookup = {
#       "h2h": h2h_dict,                    # {(team1, team2): value}
#       "team_wr": team_wr_dict,             # {team: win_rate}
#       "team_wr_venue": team_wr_venue_dict, # {(team, venue): win_rate}
#       "avg_stadium": avg_stadium_dict,     # {venue: avg_score}
#   }
#   with open("lookup.pkl", "wb") as f:
#       pickle.dump(lookup, f)

@st.cache_resource
def load_lookup():
    with open("lookup.pkl", "rb") as f:
        import pickle
        return pickle.load(f)

lookup = load_lookup()

# Clean team_wr
lookup["team_wr"] = {
    k.strip(): v
    for k, v in lookup["team_wr"].items()
}

# Clean h2h
lookup["h2h"] = {
    (k1.strip(), k2.strip()): v
    for (k1, k2), v in lookup["h2h"].items()
}

# Clean venue win rates
lookup["team_wr_venue"] = {
    (k1.strip(), venue.strip()): v
    for (k1, venue), v in lookup["team_wr_venue"].items()
}

# Clean stadium names
lookup["avg_stadium"] = {
    venue.strip(): value
    for venue, value in lookup["avg_stadium"].items()
}

# ── constants ────────────────────────────────────────────────────────────────
feature_cols = list(model.feature_names_in_)

TEAM1_OHE_COLS = sorted(
    list(
        set(
            c.replace("team1_", "").strip()
            for c in feature_cols
            if c.startswith("team1_")
        )
    )
)

TEAM2_OHE_COLS = sorted(
    list(
        set(
            c.replace("team2_", "").strip()
            for c in feature_cols
            if c.startswith("team2_")
        )
    )
)

VENUE_OHE_COLS = sorted(
    [
        c.replace("venue_", "")
        for c in feature_cols
        if c.startswith("venue_")
    ]
)

TEAMS = TEAM1_OHE_COLS
VENUES = VENUE_OHE_COLS
# ── feature builder ──────────────────────────────────────────────────────────
def build_feature_vector(
    team1, team2, venue, inning, over,
    cum_runs, cum_wickets, target_runs, target_overs,
    t1_won_toss
):
    balls = over * 6
    balls_remaining = max(0, int(target_overs) * 6 - balls)
    req_runs = max(0, target_runs - cum_runs)
    required_run_rate = (req_runs / (balls_remaining / 6)
                         if balls_remaining > 0 else 99.99)
    current_run_rate = (cum_runs / (balls / 6) if balls > 0 else 0)
    wicket_in_hands = 10 - cum_wickets

    if inning == 1:
        target_runs = 0
        req_runs = 0
        required_run_rate = 0

    # auto-fill historical features from lookup
    h2h = lookup["h2h"].get((team1, team2), 0)
    team1_win_rate = lookup["team_wr"].get(team1, 0.5)
    team2_win_rate = lookup["team_wr"].get(team2, 0.5)
    team1_wr_at_venue = lookup["team_wr_venue"].get((team1, venue), 0.5)
    team2_wr_at_venue = lookup["team_wr_venue"].get((team2, venue), 0.5)
    avg_score_of_stadium = lookup["avg_stadium"].get(venue, 165)
    
    # base features in exact column order
    row = {
        "target_runs": target_runs,
        "target_overs": target_overs,
        "h2h": h2h,
        "team1_win_rate": team1_win_rate,
        "team2_win_rate": team2_win_rate,
        "team1_wr_at_venue": team1_wr_at_venue,
        "team2_wr_at_venue": team2_wr_at_venue,
        "avg_score_of_stadium": avg_score_of_stadium,
        "inning": inning,
        "over": over,
        "ball": 6,
        "batsman_runs": 0,
        "extra_runs": 0,
        "total_runs": 0,
        "is_wicket": 0,
        "cum_wickets": cum_wickets,
        "cum_runs": cum_runs,
        "balls": balls,
        "balls_remaining": balls_remaining,
        "req_runs": req_runs,
        "required_run_rate": round(required_run_rate, 4),
        "current_run_rate": round(current_run_rate, 4),
        "wicket_in_hands": wicket_in_hands,
        "t1_won_toss": t1_won_toss,
    }

    # one-hot encode team1
    for col in TEAM1_OHE_COLS:
        row[f"team1_{col}"] = 1 if col.strip() == team1 else 0

    # one-hot encode team2
    for col in TEAM2_OHE_COLS:
        row[f"team2_{col}"] = 1 if col.strip() == team2 else 0

    # one-hot encode venue
    for col in VENUE_OHE_COLS:
        row[f"venue_{col}"] = 1 if col == venue else 0

    
    
    return pd.DataFrame([row])

# ── gauge chart ──────────────────────────────────────────────────────────────
def make_gauge(prob, team1, team2):
    pct = round(prob * 100, 1)
    color = "#3B6D11" if pct > 60 else "#A32D2D" if pct < 40 else "#854F0B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": f"{team1} win probability", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 40], "color": "#fde8e8"},
                {"range": [40, 60], "color": "#fef9e7"},
                {"range": [60, 100], "color": "#eafaf1"},
            ],
            "threshold": {
                "line": {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ── win probability curve ────────────────────────────────────────────────────
def make_prob_curve(probs_by_over, team1):
    overs = list(range(1, len(probs_by_over) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=overs, y=[p * 100 for p in probs_by_over],
        mode="lines",
        line=dict(color="royalblue", width=2),
        marker=dict(size=6),
        name=f"{team1} win %",
        hovertemplate="Over %{x}<br>Win prob: %{y:.1f}%<extra></extra>"
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="50/50")
    fig.update_layout(
        title="Win probability progression",
        xaxis_title="Over",
        yaxis_title="Win probability (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(t=40, b=40, l=40, r=20),
        hovermode="x unified"
    )
    return fig

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 Match setup")
    st.markdown("---")

    team1 = st.selectbox("Team 1 (batting first)", TEAMS, index=7)
    team2 = st.selectbox("Team 2 (chasing)", TEAMS, index=11)

    if team1 == team2:
        st.error("Team 1 and Team 2 cannot be the same.")
        st.stop()

    venue = st.selectbox("Venue", VENUES, index=14)
    inning = st.radio("Innings", [1, 2],
                      format_func=lambda x: "1st innings" if x == 1
                      else "2nd innings (chase)")
    toss_winner = st.selectbox("Toss winner", [team1, team2])
    t1_won_toss = 1 if toss_winner == team1 else 0

    st.markdown("---")
    st.markdown("### Match context (auto-filled)")
    h2h_val = lookup["h2h"].get((team1, team2), 0)
    t1wr = lookup["team_wr"].get(team1, 0.5)
    t2wr = lookup["team_wr"].get(team2, 0.5)
    t1wrv = lookup["team_wr_venue"].get((team1, venue), 0.5)
    t2wrv = lookup["team_wr_venue"].get((team2, venue), 0.5)
    avg_score = lookup["avg_stadium"].get(venue, 165)

    st.caption(f"H2H: {h2h_val}  |  Avg score here: {avg_score:.0f}")
    st.caption(f"{team1} win rate: {t1wr:.0%}  |  At venue: {t1wrv:.0%}")
    st.caption(f"{team2} win rate: {t2wr:.0%}  |  At venue: {t2wrv:.0%}")

# ── main area ────────────────────────────────────────────────────────────────
st.title("🏏 IPL Win Probability Predictor")
st.caption(f"{team1}  vs  {team2}  ·  {venue}")
st.markdown("---")

# match state inputs
col1, col2, col3 = st.columns(3)
with col1:
    over = st.slider("Completed overs", 1, 20, 10)
with col2:
    cum_runs = st.number_input("Runs scored so far", 0, 400, 85)
with col3:
    cum_wickets = st.number_input("Wickets fallen", 0, 10, 2)

col4, col5 = st.columns(2)
with col4:
    target_runs = st.number_input(
        "Target runs" if inning == 2 else "First innings runs (set at end)",
        0, 400, 180
    )
with col5:
    target_overs = st.number_input("Target overs", 1, 20, 20)

# derived values display
balls = over * 6
balls_remaining = max(0, int(target_overs) * 6 - balls)
req_runs = max(0, target_runs - cum_runs)
rrr = req_runs / (balls_remaining / 6) if balls_remaining > 0 else 99.99
crr = cum_runs / (balls / 6) if balls > 0 else 0
wih = 10 - cum_wickets

d1, d2, d3, d4 = st.columns(4)
d1.metric("Required run rate", f"{rrr:.2f}" if inning == 2 else "—")
d2.metric("Current run rate", f"{crr:.2f}")
d3.metric("Wickets in hand", wih)
d4.metric("Balls remaining", balls_remaining)

st.markdown("---")

# predict button
if st.button("⚡ Predict win probability", use_container_width=True):

    with st.spinner("Running XGBoost model..."):

        X = build_feature_vector(
            team1, team2, venue, inning, over,
            cum_runs, cum_wickets, target_runs, target_overs, t1_won_toss
        )

        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        

        # Optional safety check
        

        prob = model.predict_proba(X)[0][1]

    # results section
    st.markdown("### Result")
    gc, mc = st.columns([1, 1])

    with gc:
        st.plotly_chart(make_gauge(prob, team1, team2),
                        use_container_width=True)

    with mc:
        st.metric(f"🟢 {team1}", f"{prob:.1%}")
        st.metric(f"🔴 {team2}", f"{1-prob:.1%}")
        st.markdown("---")

        if inning == 2:
            if prob > 0.6:
                verdict = f"**{team1}** are in control. {team2} need {req_runs} runs in {balls_remaining} balls at {rrr:.2f}/over with {wih} wickets remaining."
            elif prob < 0.4:
                verdict = f"**{team2}** are favourites. They need {req_runs} in {balls_remaining} balls at {rrr:.2f}/over and have {wih} wickets in hand."
            else:
                verdict = f"Very close match. {team2} need {req_runs} in {balls_remaining} balls at {rrr:.2f}/over. {wih} wickets remaining."
        else:
            if prob > 0.6:
                verdict = f"**{team1}** are building a strong total. {cum_runs} runs in {over} overs at {crr:.2f}/over with {wih} wickets in hand."
            elif prob < 0.4:
                verdict = f"**{team1}** are slightly behind the required pace. {cum_runs} in {over} overs."
            else:
                verdict = f"A competitive 1st innings. {team1} have scored {cum_runs} in {over} overs."

        st.markdown(verdict)

    # win probability curve — run prediction for each over
    st.markdown("---")
    st.markdown("### Win probability curve")

    probs_over_time = []
    for ov in range(1, over + 1):
        frac = ov / over
        sim_runs = int(cum_runs * frac)
        sim_wickets = min(10, int(cum_wickets * frac))
        Xi = build_feature_vector(
            team1, team2, venue, inning, ov,
            sim_runs, sim_wickets, target_runs, target_overs, t1_won_toss
        )
        Xi = Xi.reindex(columns=model.feature_names_in_, fill_value=0)
        probs_over_time.append(model.predict_proba(Xi)[0][1])

    st.plotly_chart(make_prob_curve(probs_over_time, team1),
                    use_container_width=True)

    st.caption(
        "Win probability is estimated by your XGBoost model trained on "
        "historical IPL data. Not a guarantee of match outcome."
    )
    