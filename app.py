import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictCare – No-Show Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #2e6da4 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] .stRadio label { color: #ffffff !important; }

    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #2e6da4;
    }
    .card-green  { border-left-color: #27ae60; }
    .card-orange { border-left-color: #e67e22; }
    .card-red    { border-left-color: #e74c3c; }
    .card-blue   { border-left-color: #2e6da4; }
    .card-purple { border-left-color: #8e44ad; }

    /* Metric tiles */
    .metric-tile {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-tile h2 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .metric-tile p  { font-size: 0.85rem; color: #666; margin: 4px 0 0; }

    /* Risk badges */
    .badge-high   { background:#fde8e8; color:#c0392b; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }
    .badge-medium { background:#fef3e2; color:#d35400; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }
    .badge-low    { background:#e8f8f0; color:#1e8449; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }

    /* Step header */
    .step-header {
        display:flex; align-items:center; gap:12px;
        background: linear-gradient(90deg, #1a3a5c, #2e6da4);
        color: white; padding: 14px 22px;
        border-radius: 10px; margin-bottom: 16px;
    }
    .step-num {
        background:white; color:#2e6da4;
        width:32px; height:32px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-weight:700; font-size:1rem; flex-shrink:0;
    }
    .step-title { font-size:1.15rem; font-weight:600; margin:0; }
    .step-sub   { font-size:0.82rem; opacity:0.85; margin:0; }

    /* Section divider */
    .section-divider {
        border: none; border-top: 2px solid #dce6f0;
        margin: 30px 0;
    }

    /* Info box */
    .info-box {
        background:#eaf3fb; border:1px solid #aed6f1;
        border-radius:8px; padding:12px 16px;
        font-size:0.88rem; color:#1a5276;
    }

    /* Download button */
    .download-btn {
        display:inline-block;
        background:#2e6da4; color:white !important;
        padding:10px 24px; border-radius:8px;
        text-decoration:none; font-weight:600;
        font-size:0.9rem;
    }
    .download-btn:hover { background:#1a3a5c; }

    /* Table styling */
    .dataframe thead tr th { background:#2e6da4 !important; color:white !important; }
    .dataframe tbody tr:nth-child(even) { background:#f5f8fc; }

    /* Hide default header */
    header[data-testid="stHeader"] { display:none; }

    /* Top banner */
    .top-banner {
        background: linear-gradient(90deg, #1a3a5c 0%, #2e6da4 60%, #1abc9c 100%);
        color: white; padding: 22px 32px; border-radius: 12px;
        margin-bottom: 28px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .top-banner h1 { margin:0; font-size:1.9rem; font-weight:700; }
    .top-banner p  { margin:4px 0 0; opacity:0.85; font-size:0.95rem; }
    .top-logo { font-size:3rem; }

    /* Prediction table rows */
    .high-risk   td { background:#fde8e8 !important; }
    .medium-risk td { background:#fef3e2 !important; }
    .low-risk    td { background:#e8f8f0 !important; }

    /* Button hover fixes */
    .stButton > button:hover {
        background-color: #163a5c !important;
        border-color: #163a5c !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #163a5c !important;
        border-color: #163a5c !important;
    }

    /* Ensure text visibility on light backgrounds */
    .metric-tile p { color: #444444 !important; }
    .info-box { color: #1a5276 !important; }

    /* Radio button text in sidebar */
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }

    /* Secondary buttons */
    button[kind="secondary"] {
        background-color: #2e6da4 !important;
        color: #ffffff !important;
        border: none !important;
    }
    button[kind="secondary"]:hover {
        background-color: #1a3a5c !important;
    }

    /* Tab text visibility */
    .stTabs [role="tab"] {
        color: #2e6da4 !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #1a3a5c !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
for k, v in {
    "model": None,
    "label_encoders": {},
    "feature_cols": [],
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "trained": False,
    "clean_log": [],
    "train_df": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "patient_age_group",
    "previous_noshows",
    "lead_time_days",
    "appointment_type",
    "reminder_sent",
    "attended",        # target (1 = attended, 0 = no-show)
]

UPCOMING_COLS = [
    "patient_ref",
    "patient_age_group",
    "previous_noshows",
    "lead_time_days",
    "appointment_type",
    "reminder_sent",
]

FEATURE_COLS = [
    "patient_age_group",
    "previous_noshows",
    "lead_time_days",
    "appointment_type",
    "reminder_sent",
]

AGE_GROUPS   = ["Under 18", "18-30", "31-50", "51-65", "Over 65"]
APPT_TYPES   = ["New", "Follow-up"]
REMINDER_OPT = ["Yes", "No"]

RISK_LABELS  = {0: "High Risk", 1: "Medium Risk", 2: "Low Risk"}
RISK_COLORS  = {"High Risk": "#e74c3c", "Medium Risk": "#e67e22", "Low Risk": "#27ae60"}

# ─── Helper: generate sample training data ─────────────────────────────────────
def generate_sample_data(n=400):
    np.random.seed(42)
    ages   = np.random.choice(AGE_GROUPS, n)
    appt   = np.random.choice(APPT_TYPES, n)
    remind = np.random.choice(REMINDER_OPT, n, p=[0.6, 0.4])
    prev_ns = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    lead    = np.random.randint(1, 90, n)

    # Simulate attendance based on realistic rules
    attend = []
    for i in range(n):
        score = 0.7
        if prev_ns[i] >= 3:    score -= 0.4
        elif prev_ns[i] >= 1:  score -= 0.2
        if lead[i] > 30:       score -= 0.15
        if remind[i] == "No":  score -= 0.15
        if appt[i] == "New":   score -= 0.1
        if ages[i] in ["18-30", "Under 18"]: score -= 0.1
        score = max(0.05, min(0.95, score))
        attend.append(int(np.random.binomial(1, score)))

    return pd.DataFrame({
        "patient_age_group": ages,
        "previous_noshows": prev_ns,
        "lead_time_days": lead,
        "appointment_type": appt,
        "reminder_sent": remind,
        "attended": attend,
    })

# ─── Helper: generate sample upcoming data ─────────────────────────────────────
def generate_upcoming_data(n=20):
    np.random.seed(99)
    return pd.DataFrame({
        "patient_ref": [f"PT{1000+i}" for i in range(n)],
        "patient_age_group": np.random.choice(AGE_GROUPS, n),
        "previous_noshows": np.random.choice([0, 1, 2, 3], n),
        "lead_time_days": np.random.randint(1, 60, n),
        "appointment_type": np.random.choice(APPT_TYPES, n),
        "reminder_sent": np.random.choice(REMINDER_OPT, n),
    })

# ─── Helper: clean & encode training data ──────────────────────────────────────
def prepare_data(df):
    log = []
    original_len = len(df)

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, None, None, [f"❌ Missing columns: {', '.join(missing)}"]

    df = df[REQUIRED_COLS].copy()

    # Remove duplicates
    dups = df.duplicated().sum()
    if dups:
        df = df.drop_duplicates()
        log.append(f"🗑️  Removed {dups} duplicate record(s).")

    # Handle missing values
    for col in df.columns:
        na_count = df[col].isna().sum()
        if na_count:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
            log.append(f"🔧 Filled {na_count} missing value(s) in '{col}' with {'median' if df[col].dtype in [np.float64, np.int64] else 'most common value'}.")

    # Validate ranges
    before = len(df)
    df = df[df["lead_time_days"].between(0, 365)]
    df = df[df["previous_noshows"].between(0, 20)]
    removed = before - len(df)
    if removed:
        log.append(f"⚠️  Removed {removed} record(s) with out-of-range values.")

    # Validate target
    df = df[df["attended"].isin([0, 1])]

    # Encode categoricals
    le_dict = {}
    for col in ["patient_age_group", "appointment_type", "reminder_sent"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    log.append(f"✅ Data preparation complete. {len(df)} records ready for training (started with {original_len}).")

    X = df[FEATURE_COLS]
    y = df["attended"]
    return X, y, le_dict, log

# ─── Helper: encode upcoming appointments ──────────────────────────────────────
def encode_upcoming(df_up, le_dict):
    df = df_up.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in ["patient_age_group", "appointment_type", "reminder_sent"]:
        if col in le_dict:
            known = list(le_dict[col].classes_)
            df[col] = df[col].apply(lambda x: x if str(x) in known else known[0])
            df[col] = le_dict[col].transform(df[col].astype(str))

    df["previous_noshows"] = pd.to_numeric(df["previous_noshows"], errors="coerce").fillna(0).astype(int)
    df["lead_time_days"]   = pd.to_numeric(df["lead_time_days"],   errors="coerce").fillna(14).astype(int)
    return df

# ─── Helper: predict risk category ────────────────────────────────────────────
def predict_risk(model, df_encoded):
    proba = model.predict_proba(df_encoded[FEATURE_COLS])
    # proba[:,1] = probability of attending (class 1)
    # Map to risk: high(<0.5), medium(0.5-0.75), low(>=0.75)
    risk = []
    for p in proba[:, 1]:
        if p < 0.50:
            risk.append("High Risk")
        elif p < 0.75:
            risk.append("Medium Risk")
        else:
            risk.append("Low Risk")
    return risk, proba[:, 1]

# ─── Helper: get top factor ────────────────────────────────────────────────────
FACTOR_NAMES = {
    "patient_age_group": "Age Group",
    "previous_noshows":  "Previous No-Shows",
    "lead_time_days":    "Lead Time (days)",
    "appointment_type":  "Appointment Type",
    "reminder_sent":     "Reminder Sent",
}

def get_top_factor(model):
    imp = model.feature_importances_
    idx = np.argmax(imp)
    return FACTOR_NAMES[FEATURE_COLS[idx]]

# ─── Helper: CSV download link ─────────────────────────────────────────────────
def get_csv_download(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">⬇ Download {filename}</a>'

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 PredictCare")
    st.markdown("**Patient No-Show Risk Predictor**")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "🏠  Home",
        "📂  Step 1: Upload & Prepare Data",
        "🤖  Step 2: Train Model",
        "📊  Step 3: View Results",
        "🔮  Step 4: Score Upcoming",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "PredictCare uses a **Decision Tree** algorithm to classify patient appointments "
        "into High, Medium, or Low no-show risk.\n\n"
        "Built for: **BU7081 – Programming for Business Analytics**"
    )
    st.markdown("---")
    st.markdown("**Variables Used:**")
    for v in ["Patient Age Group", "Previous No-Shows", "Lead Time (days)", "Appointment Type", "Reminder Sent"]:
        st.markdown(f"• {v}")

# ─── TOP BANNER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <div>
    <h1>🏥 PredictCare</h1>
    <p>Patient No-Show Risk Prediction Platform for Outpatient Clinics</p>
  </div>
  <div class="top-logo">📋</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("## Welcome to PredictCare")
    st.markdown(
        "PredictCare helps outpatient clinic managers predict which patients are at risk of "
        "missing their appointments, enabling targeted interventions that reduce wasted slots "
        "and improve patient throughput."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-tile">
            <h2 style="color:#e74c3c;">20%</h2>
            <p>Average clinic no-show rate</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-tile">
            <h2 style="color:#2e6da4;">£200k</h2>
            <p>Annual cost per 50-slot clinic</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-tile">
            <h2 style="color:#27ae60;">5</h2>
            <p>Key predictive variables</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-tile">
            <h2 style="color:#8e44ad;">3</h2>
            <p>Risk categories: High / Med / Low</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("### How It Works")
    steps = [
        ("📂", "1. Upload Data", "Upload your historical appointment CSV or Excel file. The platform auto-cleans and validates your data."),
        ("🤖", "2. Train Model", "A Decision Tree classifier learns patterns from your historical data to identify no-show risk factors."),
        ("📊", "3. View Results", "See the decision tree diagram, model performance metrics, and risk distribution charts."),
        ("🔮", "4. Score Upcoming", "Upload tomorrow's or next week's schedule to get a risk score and top factor for every appointment."),
    ]
    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:2rem;">{icon}</div>
                <h4 style="color:#2e6da4;margin:8px 0 6px;">{title}</h4>
                <p style="font-size:0.88rem;color:#444;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Required Data Format")
    st.markdown("""
    <div class="info-box">
    Your historical appointment file must contain these columns (column names are case-insensitive):
    <br><br>
    <b>patient_age_group</b> — Under 18 / 18-30 / 31-50 / 51-65 / Over 65 &nbsp;|&nbsp;
    <b>previous_noshows</b> — Integer (0–20) &nbsp;|&nbsp;
    <b>lead_time_days</b> — Integer (days from booking to appointment) &nbsp;|&nbsp;
    <b>appointment_type</b> — New / Follow-up &nbsp;|&nbsp;
    <b>reminder_sent</b> — Yes / No &nbsp;|&nbsp;
    <b>attended</b> — 1 (attended) or 0 (no-show)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Download Sample Files")
    cc1, cc2 = st.columns(2)
    with cc1:
        sample_train = generate_sample_data(400)
        st.download_button("⬇ Sample Training Data (400 rows)", sample_train.to_csv(index=False),
                           "sample_training_data.csv", "text/csv", use_container_width=True)
    with cc2:
        sample_up = generate_upcoming_data(20)
        st.download_button("⬇ Sample Upcoming Appointments (20 rows)", sample_up.to_csv(index=False),
                           "sample_upcoming.csv", "text/csv", use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.info("👈 Use the sidebar to navigate through the four steps.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 1 – UPLOAD & PREPARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Step 1: Upload & Prepare Data":
    st.markdown("""
    <div class="step-header">
      <div class="step-num">1</div>
      <div>
        <p class="step-title">Upload & Prepare Historical Data</p>
        <p class="step-sub">Upload your appointment dataset. The system will validate, clean, and prepare it automatically.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    use_sample = st.checkbox("✅ Use built-in sample data (no upload needed — great for testing)", value=True)

    if use_sample:
        df_raw = generate_sample_data(400)
        st.success("✅ Sample dataset loaded: 400 historical appointment records.")
    else:
        uploaded = st.file_uploader(
            "Upload your historical appointment file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded is None:
            st.markdown("""
            <div class="info-box">
            📌 No file uploaded yet. Tick the box above to use sample data, or upload your own file.
            </div>""", unsafe_allow_html=True)
            st.stop()
        if uploaded.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
        st.success(f"✅ File uploaded: **{uploaded.name}** — {len(df_raw):,} rows detected.")

    st.markdown("#### Raw Data Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**Total records:** {len(df_raw):,}")
        st.markdown(f"**Columns found:** {', '.join(df_raw.columns.tolist())}")
    with col_r:
        st.markdown(f"**No-show rate:** {(1 - df_raw['attended'].mean()) * 100:.1f}%" if 'attended' in df_raw.columns else "")
        st.markdown(f"**Attended rate:** {df_raw['attended'].mean() * 100:.1f}%" if 'attended' in df_raw.columns else "")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### Data Preparation")

    if st.button("▶ Run Data Preparation", type="primary", use_container_width=True):
        with st.spinner("Preparing data..."):
            X, y, le_dict, log = prepare_data(df_raw)

        if X is None:
            for msg in log:
                st.error(msg)
        else:
            st.session_state["X_train"], st.session_state["X_test"], \
            st.session_state["y_train"], st.session_state["y_test"] = \
                train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            st.session_state["label_encoders"] = le_dict
            st.session_state["clean_log"] = log
            st.session_state["train_df"] = df_raw

            st.markdown("**Preparation Log:**")
            for msg in log:
                st.markdown(f"- {msg}")

            st.success(
                f"✅ Data ready! **{len(X)}** records prepared. "
                f"**{len(st.session_state['X_train'])}** for training, "
                f"**{len(st.session_state['X_test'])}** for validation."
            )
            st.markdown("*➡ Proceed to **Step 2: Train Model** in the sidebar.*")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 2 – TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Step 2: Train Model":
    st.markdown("""
    <div class="step-header">
      <div class="step-num">2</div>
      <div>
        <p class="step-title">Train the Decision Tree Model</p>
        <p class="step-sub">The system learns patterns from your historical data to classify appointments by no-show risk.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["X_train"] is None:
        st.warning("⚠️ Please complete Step 1 (Upload & Prepare Data) first.")
        st.stop()

    st.markdown("#### Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Maximum Tree Depth", 2, 8, 4,
                              help="Deeper trees capture more patterns but may overfit. 4 is optimal for most clinic datasets.")
        min_samples = st.slider("Minimum Samples per Leaf", 5, 30, 10,
                                help="Prevents the model from learning patterns from very small groups.")
    with col2:
        st.markdown("""
        <div class="info-box">
        <b>Recommended settings:</b><br>
        • Max Depth: 4 (balances accuracy and interpretability)<br>
        • Min Samples: 10 (prevents overfitting on small groups)<br><br>
        The model will be evaluated on a held-out 20% validation set.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if st.button("▶ Train Decision Tree Model", type="primary", use_container_width=True):
        with st.spinner("Training model... please wait."):
            import time
            time.sleep(0.5)  # brief pause for UX

            X_train = st.session_state["X_train"]
            X_test  = st.session_state["X_test"]
            y_train = st.session_state["y_train"]
            y_test  = st.session_state["y_test"]

            # Build 3-class target: 0=high risk, 1=medium, 2=low
            # Map from binary attended to 3 class using probability buckets after training
            # First train binary model
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples,
                random_state=42,
                class_weight="balanced",
            )
            clf.fit(X_train, y_train)

            st.session_state["model"]   = clf
            st.session_state["trained"] = True

        y_pred  = clf.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        cm      = confusion_matrix(y_test, y_pred)

        # Sensitivity & specificity (no-show = class 0)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        st.success("✅ Model trained successfully!")

        st.markdown("#### Validation Performance")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-tile"><h2 style="color:#2e6da4;">{acc*100:.1f}%</h2><p>Overall Accuracy</p></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-tile"><h2 style="color:#e74c3c;">{sensitivity*100:.1f}%</h2><p>Sensitivity (No-Show Detection)</p></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-tile"><h2 style="color:#27ae60;">{specificity*100:.1f}%</h2><p>Specificity (Attender Detection)</p></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-tile"><h2 style="color:#8e44ad;">{len(X_test)}</h2><p>Validation Records Used</p></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:16px;">
        <b>How to read these metrics:</b><br>
        <b>Overall Accuracy</b> – the proportion of all appointments correctly classified as attended or no-show.<br>
        <b>Sensitivity</b> – of all actual no-shows, what proportion were correctly identified as high/medium risk.<br>
        <b>Specificity</b> – of all patients who actually attended, what proportion were correctly classified as low risk.
        </div>""", unsafe_allow_html=True)

        st.markdown("*➡ Proceed to **Step 3: View Results** in the sidebar.*")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 3 – RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Step 3: View Results":
    st.markdown("""
    <div class="step-header">
      <div class="step-num">3</div>
      <div>
        <p class="step-title">Model Results & Analysis</p>
        <p class="step-sub">Explore the decision tree, performance metrics, and risk distribution across your patient population.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["trained"]:
        st.warning("⚠️ Please complete Step 2 (Train Model) first.")
        st.stop()

    model   = st.session_state["model"]
    X_train = st.session_state["X_train"]
    X_test  = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌳 Decision Tree", "📈 Performance", "📊 Risk Distribution", "🔑 Feature Importance"
    ])

    # ── TAB 1: Tree Diagram ──
    with tab1:
        st.markdown("#### Decision Tree Structure")
        st.markdown("""
        <div class="info-box">
        Each node shows the variable and threshold used to split the data.
        Left branches follow the condition being <b>True</b>; right branches follow <b>False</b>.
        Leaf nodes show the predicted outcome and the sample count.
        </div>""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            model,
            feature_names=list(FACTOR_NAMES.values()),
            class_names=["No-Show", "Attended"],
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
            impurity=False,
            proportion=True,
        )
        ax.set_facecolor("#f8fbff")
        fig.patch.set_facecolor("#f8fbff")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Decision Rules (Text Format)")
        with st.expander("Click to expand full rule set"):
            rules = export_text(model, feature_names=list(FACTOR_NAMES.values()))
            st.code(rules, language="")

    # ── TAB 2: Performance ──
    with tab2:
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Confusion Matrix")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(cm, display_labels=["No-Show", "Attended"])
            disp.plot(ax=ax2, colorbar=False, cmap="Blues")
            ax2.set_title("Validation Set Confusion Matrix", fontsize=12, fontweight="bold")
            fig2.patch.set_facecolor("white")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("#### Performance Summary")
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
            metrics = {
                "Overall Accuracy":        f"{acc*100:.1f}%",
                "Sensitivity (Recall)":    f"{tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "N/A",
                "Specificity":             f"{tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "N/A",
                "Precision":               f"{tp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else "N/A",
                "True No-Shows Detected":  str(tp),
                "Missed No-Shows":         str(fn),
                "Validation Set Size":     str(len(y_test)),
            }
            for k, v in metrics.items():
                c_l, c_r = st.columns([2, 1])
                c_l.markdown(f"**{k}**")
                c_r.markdown(f"`{v}`")

            st.markdown("""
            <div class="info-box" style="margin-top:12px;">
            <b>Sensitivity</b> is the most important metric for this use case:
            it tells you what proportion of actual no-shows the model successfully flagged as high or medium risk.
            A sensitivity above 70% means the model catches the majority of no-shows for proactive intervention.
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: Risk Distribution ──
    with tab3:
        st.markdown("#### Risk Category Distribution — Training Data")
        _, proba_train = predict_risk(model, X_train)
        risk_train = []
        for p in proba_train:
            if p < 0.50:   risk_train.append("High Risk")
            elif p < 0.75: risk_train.append("Medium Risk")
            else:          risk_train.append("Low Risk")

        risk_counts = pd.Series(risk_train).value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)

        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map=RISK_COLORS,
                hole=0.45,
                title="Patient Risk Distribution",
            )
            fig3.update_traces(textposition="outside", textinfo="percent+label")
            fig3.update_layout(showlegend=True, font=dict(family="Arial", size=13),
                               title_font_size=15, margin=dict(t=50, b=20))
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            fig4 = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                color=risk_counts.index,
                color_discrete_map=RISK_COLORS,
                title="Risk Category Counts",
                labels={"x": "Risk Category", "y": "Number of Appointments"},
                text=risk_counts.values,
            )
            fig4.update_traces(textposition="outside")
            fig4.update_layout(showlegend=False, font=dict(family="Arial", size=13),
                               title_font_size=15, xaxis_title="", yaxis_title="Count",
                               margin=dict(t=50, b=20))
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("#### Recommended Actions by Risk Level")
        action_data = {
            "Risk Level": ["🔴 High Risk", "🟠 Medium Risk", "🟢 Low Risk"],
            "No-Show Probability": ["> 50%", "25% – 50%", "< 25%"],
            "Recommended Action": [
                "Phone call reminder + consider double-booking slot",
                "SMS reminder 48 hrs before; offer rebooking option",
                "Standard automated reminder only",
            ],
            "Priority": ["Immediate action", "Moderate attention", "Standard process"],
        }
        st.dataframe(pd.DataFrame(action_data), use_container_width=True, hide_index=True)

    # ── TAB 4: Feature Importance ──
    with tab4:
        st.markdown("#### Feature Importance — What Drives No-Show Risk?")
        importance_df = pd.DataFrame({
            "Variable": list(FACTOR_NAMES.values()),
            "Importance Score": model.feature_importances_,
        }).sort_values("Importance Score", ascending=True)

        fig5 = px.bar(
            importance_df, x="Importance Score", y="Variable",
            orientation="h",
            title="Decision Tree Feature Importance",
            color="Importance Score",
            color_continuous_scale=["#aed6f1", "#2e6da4"],
            text=importance_df["Importance Score"].apply(lambda x: f"{x:.3f}"),
        )
        fig5.update_traces(textposition="outside")
        fig5.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis_title="Relative Importance", yaxis_title="",
            font=dict(family="Arial", size=13), title_font_size=15,
            margin=dict(l=10, r=30, t=50, b=20),
        )
        st.plotly_chart(fig5, use_container_width=True)

        top_feat = FACTOR_NAMES[FEATURE_COLS[np.argmax(model.feature_importances_)]]
        st.markdown(f"""
        <div class="info-box">
        🔑 <b>Most important predictor in your data: {top_feat}</b><br>
        Higher importance scores indicate that the variable plays a larger role in distinguishing
        patients who attend from those who do not. Variables with very low importance (near zero)
        contribute little to the classification and could be considered for removal in future iterations.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 4 – SCORE UPCOMING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Step 4: Score Upcoming":
    st.markdown("""
    <div class="step-header">
      <div class="step-num">4</div>
      <div>
        <p class="step-title">Score Upcoming Appointments</p>
        <p class="step-sub">Upload your schedule to receive a no-show risk category and top contributing factor for each appointment.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["trained"]:
        st.warning("⚠️ Please complete Steps 1 and 2 first.")
        st.stop()

    st.markdown("""
    <div class="info-box">
    Upload a file containing your upcoming appointments. Required columns:
    <b>patient_ref</b>, <b>patient_age_group</b>, <b>previous_noshows</b>,
    <b>lead_time_days</b>, <b>appointment_type</b>, <b>reminder_sent</b>
    (no 'attended' column needed).
    </div>""", unsafe_allow_html=True)

    use_sample_up = st.checkbox("✅ Use sample upcoming appointments (20 records)", value=True)

    if use_sample_up:
        df_up_raw = generate_upcoming_data(20)
        st.success("✅ Sample upcoming schedule loaded: 20 appointments.")
    else:
        uploaded_up = st.file_uploader(
            "Upload upcoming appointment schedule (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="upcoming_uploader",
        )
        if uploaded_up is None:
            st.stop()
        df_up_raw = pd.read_csv(uploaded_up) if uploaded_up.name.endswith(".csv") else pd.read_excel(uploaded_up)
        st.success(f"✅ Loaded {len(df_up_raw)} upcoming appointments.")

    st.markdown("#### Uploaded Schedule Preview")
    st.dataframe(df_up_raw.head(10), use_container_width=True)

    if st.button("▶ Generate Risk Predictions", type="primary", use_container_width=True):
        model   = st.session_state["model"]
        le_dict = st.session_state["label_encoders"]

        df_encoded = encode_upcoming(df_up_raw, le_dict)
        risk_labels, proba = predict_risk(model, df_encoded)
        top_factor = get_top_factor(model)

        result_df = df_up_raw.copy()
        result_df["Risk Category"]       = risk_labels
        result_df["Attend Probability"]  = [f"{p*100:.0f}%" for p in proba]
        result_df["Top Risk Factor"]     = top_factor
        result_df["Recommended Action"]  = result_df["Risk Category"].map({
            "High Risk":   "📞 Phone call + consider double-booking",
            "Medium Risk": "📱 SMS reminder 48hrs before",
            "Low Risk":    "✉️ Standard automated reminder",
        })

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Risk Prediction Results")

        # Summary tiles
        counts = pd.Series(risk_labels).value_counts()
        high   = counts.get("High Risk",   0)
        medium = counts.get("Medium Risk", 0)
        low    = counts.get("Low Risk",    0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-tile card card-red"><h2 style="color:#e74c3c;">{high}</h2><p>🔴 High Risk Appointments</p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-tile card card-orange"><h2 style="color:#e67e22;">{medium}</h2><p>🟠 Medium Risk Appointments</p></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-tile card card-green"><h2 style="color:#27ae60;">{low}</h2><p>🟢 Low Risk Appointments</p></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk distribution bar
        fig_up = px.bar(
            x=["High Risk", "Medium Risk", "Low Risk"],
            y=[high, medium, low],
            color=["High Risk", "Medium Risk", "Low Risk"],
            color_discrete_map=RISK_COLORS,
            title=f"Risk Distribution — {len(result_df)} Upcoming Appointments",
            text=[high, medium, low],
        )
        fig_up.update_traces(textposition="outside")
        fig_up.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count",
                             font=dict(family="Arial", size=13), margin=dict(t=50, b=20))
        st.plotly_chart(fig_up, use_container_width=True)

        # Full results table
        st.markdown("#### Full Results Table (sorted by risk)")
        risk_order = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}
        result_df_sorted = result_df.sort_values("Risk Category", key=lambda x: x.map(risk_order))

        def highlight_risk(row):
            colour_map = {
                "High Risk":   "background-color: #fde8e8",
                "Medium Risk": "background-color: #fef3e2",
                "Low Risk":    "background-color: #e8f8f0",
            }
            return [colour_map.get(row["Risk Category"], "")] * len(row)

        styled = result_df_sorted.style.apply(highlight_risk, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### Download Results")
        st.download_button(
            "⬇ Download Full Risk Report (CSV)",
            result_df_sorted.to_csv(index=False),
            "predictcare_risk_report.csv",
            "text/csv",
            use_container_width=True,
        )
        st.markdown("""
        <div class="info-box" style="margin-top:16px;">
        💡 <b>Next steps:</b> Share this report with reception staff before the appointment day.
        For all High Risk slots, initiate a phone call reminder. Consider adding waitlisted patients
        to High Risk slots if cancellations are not received 24 hours before the appointment.
        </div>""", unsafe_allow_html=True)
