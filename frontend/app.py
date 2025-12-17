import os
import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Premier League - Classement Final",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CONFIGURATION MLFLOW ------------------
# Set MLflow tracking URI and credentials
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/7a99ui/premier_league_mlops.mlflow'

# Check if credentials are provided via environment variables
if os.getenv('MLFLOW_TRACKING_USERNAME') and os.getenv('MLFLOW_TRACKING_PASSWORD'):
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
    st.sidebar.info("🔐 Authentification via variables d'environnement")
else:
    st.sidebar.warning("⚠️ Credentials MLflow non trouvés. Configurez MLFLOW_TRACKING_USERNAME et MLFLOW_TRACKING_PASSWORD")
    st.error("""
    ❌ **Configuration requise:**
    
    Veuillez définir les variables d'environnement suivantes dans votre docker-compose.yml:
    
    ```yaml
    environment:
      - MLFLOW_TRACKING_USERNAME=votre_username
      - MLFLOW_TRACKING_PASSWORD=votre_token
    ```
    
    Obtenez votre token sur: https://dagshub.com/user/settings/tokens
    """)
    st.stop()

# ------------------ MODERN CSS THEME ------------------
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Dark Theme Background */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: #0f0c29;
    }
    
    /* Modern Header Card */
    .header-card {
        background: linear-gradient(135deg, rgba(88, 86, 214, 0.1) 0%, rgba(131, 58, 180, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .pl-logo {
        width: 140px;
        height: auto;
        margin-bottom: 1rem;
        background: white;
        padding: 1rem;
        border-radius: 16px;
        filter: drop-shadow(0 8px 20px rgba(0, 0, 0, 0.4));
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.7);
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .season-pill {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.6rem 1.8rem;
        border-radius: 100px;
        font-weight: 700;
        margin-top: 1rem;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Modern Table Container */
    .modern-table-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Clean Table Styles */
    .stDataFrame, .stTable {
        width: 100%;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    /* Modern Table Header */
    thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 1.2rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: none !important;
    }
    
    /* Table Body */
    tbody td {
        padding: 1.2rem 1.5rem !important;
        font-size: 1rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: rgba(255, 255, 255, 0.95) !important;
        background: transparent !important;
        transition: all 0.2s ease;
    }
    
    tbody tr {
        background: transparent !important;
        transition: all 0.2s ease;
    }
    
    tbody tr:hover {
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    tbody tr:hover td {
        background: transparent !important;
    }
    
    /* Position Column - Modern Badges */
    tbody td:first-child {
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        width: 60px !important;
        max-width: 60px !important;
    }
    
    /* Gold for 1st place */
    tbody tr:nth-child(1) td:first-child {
        color: #FFD700 !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    /* Silver for 2nd place */
    tbody tr:nth-child(2) td:first-child {
        color: #C0C0C0 !important;
        text-shadow: 0 0 10px rgba(192, 192, 192, 0.5);
    }
    
    /* Bronze for 3rd place */
    tbody tr:nth-child(3) td:first-child {
        color: #CD7F32 !important;
        text-shadow: 0 0 10px rgba(205, 127, 50, 0.5);
    }
    
    /* Top 4 - Champions League Indicator */
    tbody tr:nth-child(-n+4) {
        border-left: 4px solid #667eea !important;
        background: rgba(102, 126, 234, 0.05) !important;
    }
    
    /* Relegation Zone */
    tbody tr:nth-child(n+18) {
        border-left: 4px solid #ff6b6b !important;
        background: rgba(255, 107, 107, 0.05) !important;
    }
    
    /* Team Names - Bold */
    tbody td:nth-child(2) {
        font-weight: 600 !important;
        color: white !important;
    }
    
    /* Points Column - Highlighted */
    tbody td:nth-child(3) {
        font-weight: 700 !important;
        color: #667eea !important;
        font-size: 1.1rem !important;
    }
    
    /* Modern Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #667eea !important;
        font-weight: 700 !important;
    }
    
    /* Modern Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stat-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .stat-card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-card-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .stat-card-team {
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .stat-card-points {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 800;
    }
    
    .stat-card-value {
        color: #667eea;
        font-size: 2rem;
        font-weight: 800;
    }
    
    /* Modern Legend Cards */
    .legend-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .legend-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .legend-indicator {
        width: 4px;
        height: 40px;
        border-radius: 2px;
    }
    
    .legend-text {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Alert Styles */
    .stAlert {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px;
        color: white !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------ PATHS ------------------
DATA_PATH = Path("data/processed/test.parquet")

# ------------------ LOAD DATA ------------------
try:
    data = pd.read_parquet(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ Fichier introuvable : {DATA_PATH}")
    st.stop()

# ------------------ LOAD MODEL FROM MLFLOW ------------------
@st.cache_resource
def load_model_from_mlflow():
    """Charge le modèle depuis MLflow Registry"""
    try:
        client = MlflowClient()
        
        # Récupérer toutes les versions du modèle
        versions = client.search_model_versions("name='PremierLeagueModel'")
        
        if not versions:
            st.error("❌ Aucune version du modèle PremierLeagueModel trouvée dans MLflow")
            st.stop()
        
        # Chercher la version avec le tag deployment_status=production
        production_version = None
        for v in versions:
            if v.tags.get('deployment_status') == 'production':
                production_version = v.version
                break
        
        # Si aucune version n'a le tag, prendre la dernière version
        if not production_version:
            production_version = versions[0].version
            st.sidebar.warning(f"⚠️ Aucun modèle en production, utilisation de la version {production_version}")
        
        # Charger le modèle
        model_uri = f"models:/PremierLeagueModel/{production_version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Afficher les infos du modèle dans la sidebar
        st.sidebar.success(f"✅ Modèle v{production_version} chargé depuis MLflow")
        
        # Afficher les métriques si disponibles
        model_version_details = client.get_model_version("PremierLeagueModel", production_version)
        if model_version_details.tags:
            st.sidebar.markdown("**📊 Métriques du modèle:**")
            if 'val_mae' in model_version_details.tags:
                st.sidebar.text(f"Val MAE: {model_version_details.tags['val_mae']}")
            if 'test_mae' in model_version_details.tags:
                st.sidebar.text(f"Test MAE: {model_version_details.tags['test_mae']}")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle depuis MLflow: {e}")
        st.stop()

model = load_model_from_mlflow()

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("### ⚙️ PARAMÈTRES")
st.sidebar.markdown("---")

available_seasons = sorted(data["season"].unique())
selected_season = st.sidebar.selectbox(
    "🏆 Sélectionner la saison",
    available_seasons
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 À PROPOS")
st.sidebar.info(
    "Cette application prédit le classement final de la Premier League "
    "en utilisant un modèle de Machine Learning entraîné sur les données historiques."
)

# ------------------ HEADER ------------------
st.markdown(f"""
<div class="header-card">
    <img src="https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg" alt="Premier League" class="pl-logo">
    <h1 class="header-title">Premier League</h1>
    <p class="header-subtitle">Classement Final Prédit par Machine Learning</p>
    <div class="season-pill">Saison {selected_season}</div>
</div>
""", unsafe_allow_html=True)

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

# ------------------ DISPLAY TABLE ------------------
display_table = (
    final_table[["Position", "team", "predicted_points"]]
    .rename(columns={
        "team": "Équipe",
        "predicted_points": "Points"
    })
)

display_table["Points"] = display_table["Points"].round(0).astype(int)

# ------------------ STYLED DISPLAY ------------------
st.markdown('<div class="modern-table-container">', unsafe_allow_html=True)

# Utiliser st.dataframe avec height pour afficher toutes les lignes
st.dataframe(
    display_table,
    hide_index=True,
    use_container_width=True,
    height=730,  # Hauteur suffisante pour 20 équipes
    column_config={
        "Position": st.column_config.NumberColumn(
            "Position",
            help="Classement",
            width="small"
        ),
        "Équipe": st.column_config.TextColumn(
            "Équipe",
            help="Nom de l'équipe",
            width="medium"
        ),
        "Points": st.column_config.NumberColumn(
            "Points",
            help="Points prédits",
            width="small"
        )
    }
)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ STATS CARDS ------------------
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-card-icon">🏆</div>
        <div class="stat-card-label">Champion</div>
        <div class="stat-card-team">{display_table.iloc[0]['Équipe']}</div>
        <div class="stat-card-points">{display_table.iloc[0]['Points']} pts</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-card-icon">📉</div>
        <div class="stat-card-label">Dernier</div>
        <div class="stat-card-team">{display_table.iloc[-1]['Équipe']}</div>
        <div class="stat-card-points">{display_table.iloc[-1]['Points']} pts</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    point_diff = display_table.iloc[0]['Points'] - display_table.iloc[1]['Points']
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-card-icon">📊</div>
        <div class="stat-card-label">Écart 1er-2e</div>
        <div class="stat-card-value">{point_diff} pts</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_points = display_table['Points'].mean()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-card-icon">⚡</div>
        <div class="stat-card-label">Moyenne</div>
        <div class="stat-card-value">{avg_points:.1f} pts</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------ LEGEND ------------------
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="legend-card">
        <div class="legend-item">
            <div class="legend-indicator" style="background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);"></div>
            <div class="legend-text"><strong>Top 4</strong><br>Champions League</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="legend-card">
        <div class="legend-item">
            <div class="legend-indicator" style="background: linear-gradient(180deg, #FFD700 0%, #FFA500 100%);"></div>
            <div class="legend-text"><strong>1er Place</strong><br>Champion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="legend-card">
        <div class="legend-item">
            <div class="legend-indicator" style="background: linear-gradient(180deg, #ff6b6b 0%, #ee5a6f 100%);"></div>
            <div class="legend-text"><strong>18-20</strong><br>Relégation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------ SANITY CHECK ------------------
max_points = display_table["Points"].max()

st.markdown("<br>", unsafe_allow_html=True)

if max_points > 114:
    st.warning(
        "⚠️ **Attention:** Points supérieurs au maximum théorique (114). "
        "Vérifiez la target ou le modèle."
    )