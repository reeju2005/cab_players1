import streamlit as st
import pandas as pd
import numpy as np
import os, glob, re

st.set_page_config(page_title="CAB Prediction Studio", layout="wide")

st.title("🏏 CAB Prediction Studio")
st.caption("Predict team of the year, players to watch, and next-season performance. Add more team CSVs anytime.")

# Default folder set to CAB_AllCompetitions
default_data_folder = "CAB_AllCompetitions"

@st.cache_data
def load_data(folder):
    paths = sorted(glob.glob(os.path.join(folder, "CAB - *.csv")))
    frames = []
    info = []
    for p in paths:
        team = os.path.basename(p).replace("CAB - ", "").replace(".csv", "").strip()
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.read_csv(p, encoding="latin-1")
        df["__Team"] = team
        frames.append(df)
        info.append({"file": os.path.basename(p), "team": team, "rows": len(df), "cols": len(df.columns)})
    data = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return data, pd.DataFrame(info)

data_folder = st.sidebar.text_input("Data folder", default_data_folder, help="Folder containing files named like: CAB - TEAM.csv")
data, files_info = load_data(data_folder)

if files_info.empty:
    st.warning("No team CSVs found in the specified folder. Place files like 'CAB - TEAM.csv' in CAB_AllCompetitions.")
    st.stop()

with st.expander("📂 Loaded files", expanded=False):
    st.dataframe(files_info, use_container_width=True)

def normalize_columns(df):
    cols = {}
    for c in df.columns:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", " ", c2)
        cols[c] = c2
    return df.rename(columns=cols)

data = normalize_columns(data)

all_cols = [c for c in data.columns if c != "__Team"]
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
string_cols = data.select_dtypes(include=["object"]).columns.tolist()

def pick_column(candidates, preferred):
    lc = {c: c.lower() for c in candidates}
    for p in preferred:
        for c in candidates:
            if p in lc[c]:
                return c
    return None

player_col = pick_column(all_cols, ["player", "name", "batsman", "batter", "bowler"])
season_col = pick_column(all_cols, ["season", "year", "yr"])
team_col = "__Team"

st.sidebar.markdown("### Columns Detected")
st.sidebar.write("**Player:** ", player_col or "—")
st.sidebar.write("**Season/Year:** ", season_col or "—")
st.sidebar.write("**Team:** ", team_col)

def composite_score(df, metrics, weights):
    if not metrics:
        return pd.Series([np.nan] * len(df), index=df.index, name="__Score")
    total = None
    for m in metrics:
        w = weights.get(m, 1.0)
        z = (df[m] - df[m].mean()) / df[m].std(ddof=0) if df[m].std(ddof=0) not in (0, np.nan) else df[m].fillna(0)
        part = w * z
        total = part if total is None else total + part
    if total is None:
        return pd.Series([np.nan] * len(df), index=df.index, name="__Score")
    return total.rename("__Score")

metrics = st.multiselect("Select performance metric(s) to optimize", options=numeric_cols, default=numeric_cols[:1])

weights = {}
if metrics:
    st.markdown("### ⚖️ Weight your metrics")
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with cols[i]:
            weights[m] = st.number_input(f"Weight for **{m}**", value=1.0, step=0.1)

work = data.copy()
for m in metrics:
    if m in work.columns:
        work[m] = pd.to_numeric(work[m], errors="coerce")

work["__Score"] = composite_score(work, metrics, weights)

st.markdown("## 🏆 Team of the Year")
k_players = st.slider("Top-k players considered per team", min_value=3, max_value=15, value=11, step=1)
method = st.selectbox("Ranking method", ["Average of top-k scores", "Sum of top-k scores"])

def rank_teams(df, k=11, method="Average of top-k scores"):
    if "__Score" not in df.columns or df["__Score"].isna().all():
        return pd.DataFrame()
    parts = []
    for team, g in df.groupby("__Team"):
        g = g.sort_values("__Score", ascending=False)
        topk = g.head(k)
        score = topk["__Score"].mean() if method.startswith("Average") else topk["__Score"].sum()
        parts.append({"Team": team, "TeamScore": score})
    res = pd.DataFrame(parts).sort_values("TeamScore", ascending=False, ignore_index=True)
    return res

team_table = rank_teams(work, k_players, method)
if team_table.empty:
    st.warning("Cannot compute team ranks — select at least one numeric metric.")
else:
    st.dataframe(team_table, use_container_width=True)
    if st.checkbox("Show Team of the Year lineup (Top XI overall)"):
        top_xi = work.sort_values("__Score", ascending=False).head(11)
        st.dataframe(top_xi[[player_col, "__Team"] + metrics] if player_col else top_xi[["__Team"] + metrics], use_container_width=True)

st.markdown("## 🔮 Predict Next Season Performance")

def predict_next_season(df, player_col, season_col, metric):
    res = []
    if season_col and player_col:
        for (p, t), g in df[[player_col, "__Team", season_col, metric]].dropna().groupby([player_col, "__Team"]):
            if g[season_col].nunique() < 2:
                continue
            try:
                x = pd.to_numeric(g[season_col], errors="coerce").values.astype(float)
                y = g[metric].values.astype(float)
                b1, b0 = np.polyfit(x, y, 1)
                next_season = np.nanmax(x) + 1
                yhat = b1 * next_season + b0
                res.append({"Player": p, "Team": t, "Current": y[-1], "PredictedNext": float(yhat), "TrendSlope": float(b1)})
            except Exception:
                continue
    else:
        mu_team = df.groupby("__Team")[metric].transform("mean")
        mu_global = df[metric].mean()
        alpha = 0.7
        pred = alpha * df[metric] + (1 - alpha) * ((mu_team + mu_global) / 2)
        for i, row in df.iterrows():
            res.append({"Player": row.get(player_col, f"Player#{i}"), "Team": row["__Team"], "Current": row[metric], "PredictedNext": pred.loc[i], "TrendSlope": np.nan})
    return pd.DataFrame(res)

if metrics:
    target_metric = st.selectbox("Target metric for prediction", metrics, index=0)
    preds = predict_next_season(work, player_col, season_col, target_metric)
    if preds.empty:
        preds = predict_next_season(work, player_col, None, target_metric)
    st.dataframe(preds.sort_values("PredictedNext", ascending=False), use_container_width=True)

    st.markdown("### 🌟 Players to Watch")
    topn = st.slider("How many players?", 5, 30, 11, 1)
    if "TrendSlope" in preds.columns and preds["TrendSlope"].notna().any():
        watch = preds.sort_values(["TrendSlope", "PredictedNext"], ascending=[False, False]).head(topn)
    else:
        watch = preds.sort_values("PredictedNext", ascending=False).head(topn)
    st.dataframe(watch, use_container_width=True)

st.markdown("---")
st.markdown("#### ➕ Add more teams later")
st.write("Just drop another CSV named like **CAB - TEAMNAME.csv** into the CAB_AllCompetitions folder and reload the app.")
