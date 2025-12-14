# frontend/app.py

import streamlit as st
import pandas as pd
from joblib import load
from pathlib import Path

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Premier League - Classement Final",
    layout="wide"
)

st.title("🏆 Premier League – Classement Final Prédit")

# ------------------ PATHS (DOCKER FRIENDLY) ------------------
DATA_PATH = Path("data/processed/v1/test.parquet")
MODEL_PATH = Path("models/production/latest_model.joblib")

# ------------------ LOAD DATA ------------------
try:
    data = pd.read_parquet(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ Fichier introuvable : {DATA_PATH}")
    st.stop()

# ------------------ LOAD MODEL ------------------
try:
    model = load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"❌ Modèle introuvable : {MODEL_PATH}")
    st.stop()

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Paramètres")

available_seasons = sorted(data["season"].unique())
selected_season = st.sidebar.selectbox(
    "Sélectionner la saison",
    available_seasons
)

# ------------------ FILTER SEASON ------------------
season_data = data[data["season"] == selected_season].copy()

# ------------------ FEATURE SELECTION ------------------
metadata_cols = ["season", "team", "gameweek"]
target_col = "target_final_points"

feature_cols = [
    col for col in season_data.columns
    if col not in metadata_cols + [target_col]
]

X = season_data[feature_cols]

# ------------------ PREDICTION ------------------
season_data["predicted_points"] = model.predict(X)

# ------------------ DERNIÈRE GAMEWEEK ------------------
last_gameweek = season_data["gameweek"].max()

final_gw_data = season_data[
    season_data["gameweek"] == last_gameweek
].copy()

# ------------------ FINAL TABLE ------------------
final_table = (
    final_gw_data[["team", "predicted_points"]]
    .sort_values(by="predicted_points", ascending=False)
    .reset_index(drop=True)
)

final_table["Position"] = final_table.index + 1

# ------------------ DISPLAY TABLE (3 COLONNES) ------------------
display_table = (
    final_table[["Position", "team", "predicted_points"]]
    .rename(columns={
        "team": "Équipe",
        "predicted_points": "Points"
    })
)

display_table["Points"] = display_table["Points"].round(0).astype(int)

# ------------------ STYLED DISPLAY ------------------
st.subheader(f"📊 Classement final prédit – Saison {selected_season}")

styled_table = (
    display_table
    .style
    .hide(axis="index")  # ✅ SUPPRIME L’INDEX (PAS DE 4ᵉ COLONNE)
    .set_properties(
        **{
            "text-align": "center",
            "vertical-align": "middle",
            "font-size": "16px"
        }
    )
    .set_table_styles([
        {"selector": "th", "props": [
            ("text-align", "center"),
            ("font-size", "17px")
        ]}
    ])
)

st.table(styled_table)

# ------------------ SANITY CHECK ------------------
max_points = display_table["Points"].max()

if max_points > 114:
    st.warning(
        "⚠️ Attention : points > maximum théorique (114). "
        "Vérifiez la target ou le modèle."
    )
else:
    st.success("✅ Points cohérents avec une saison de Premier League.")
