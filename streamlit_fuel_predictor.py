import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import tempfile
import os
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ------------------ Configuration (repo-root paths) ------------------
DATA_XLSX = "diesel_properties_clean.xlsx"
SPEC_XLSX = "diesel_spec.xlsx"  # optional
SCALER_PATH = "scaler.joblib"
PLS_PATH = "pls_model.joblib"
RF_ZIP_PATH = "rf_model.zip"
# ------------------------------------------------------------------

st.set_page_config(page_title="Fuel Parameter Predictor", layout="wide")

# Gentle, readable theme (teal accents on light/dark neutral background)
st.markdown(
    """
    <style>
    :root { --bg: #f6f8fa; --card: #ffffff; --text: #0f1724; --muted: #475569; --accent: #0ea5a4; }
    .stApp { background: var(--bg); color: var(--text); }
    .stSidebar .stButton>button { background-color: var(--accent); color: white; }
    .stButton>button { background-color: var(--accent); color: white; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { background: #fff; color: var(--text); }
    .stMarkdown { color: var(--text); }
    .small-muted { color: var(--muted); font-size:12px; margin-top:-8px }
    table { color: var(--text); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fuel Parameter Predictor")
st.write("Enter **any two** numeric parameters below. Ranges are shown under each input to help you.")

# ------------------ Helper functions ------------------

def load_dataset(path=DATA_XLSX):
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}. Make sure you uploaded `{path}` to your repo root.")
        st.stop()
    df = pd.read_excel(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for c in ["ID", "Sample", "SampleID", "index", "Unnamed: 0"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != 'label']
    df = df[numeric_cols].copy()
    if df.shape[1] == 0:
        st.error("No numeric columns found in dataset. Please check the uploaded Excel file.")
        st.stop()
    return df


def build_imputer(df):
    imp = IterativeImputer(random_state=0, sample_posterior=False, max_iter=20)
    imp.fit(df.values)
    return imp


def extract_models_from_zip(zip_path):
    models = {}
    if not os.path.exists(zip_path):
        return models
    try:
        tmpdir = tempfile.mkdtemp(prefix="rf_models_")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        for p in Path(tmpdir).rglob("*"):
            if p.suffix.lower() in ('.joblib', '.pkl'):
                try:
                    m = joblib.load(p.as_posix())
                    models[p.name] = m
                except Exception:
                    pass
    except zipfile.BadZipFile:
        st.warning("rf_model.zip is not a valid zip file or is missing.")
    return models


def create_input_row(feature_names, user_inputs):
    arr = np.full((1, len(feature_names)), np.nan, dtype=float)
    for i, fname in enumerate(feature_names):
        if fname in user_inputs:
            arr[0, i] = float(user_inputs[fname])
    return arr


def apply_model_predictions(model, X_for_models, feature_names, current_result, user_inputs, name_hint=None):
    sources = {}
    try:
        ypred = model.predict(X_for_models)
    except Exception:
        return current_result, sources

    ypred = np.asarray(ypred)
    n_features = len(feature_names)

    if ypred.ndim == 2 and ypred.shape[1] == n_features:
        for i, fname in enumerate(feature_names):
            if fname not in user_inputs:
                current_result[fname] = float(ypred[0, i])
                sources[fname] = name_hint or 'model'
    elif ypred.ndim == 1:
        k = ypred.shape[0]
        if k == n_features:
            for i, fname in enumerate(feature_names):
                if fname not in user_inputs:
                    current_result[fname] = float(ypred[i])
                    sources[fname] = name_hint or 'model'
        elif k == 1:
            if name_hint:
                base = Path(name_hint).stem
                if base in feature_names and base not in user_inputs:
                    current_result[base] = float(ypred[0])
                    sources[base] = name_hint
            else:
                missing = [f for f in feature_names if f not in user_inputs]
                if missing:
                    current_result[missing[0]] = float(ypred[0])
                    sources[missing[0]] = name_hint or 'model'
        else:
            missing = [f for f in feature_names if f not in user_inputs]
            for j in range(min(k, len(missing))):
                current_result[missing[j]] = float(ypred[j])
                sources[missing[j]] = name_hint or 'model'
    else:
        missing = [f for f in feature_names if f not in user_inputs]
        flat = ypred.ravel()
        for j in range(min(len(flat), len(missing))):
            current_result[missing[j]] = float(flat[j])
            sources[missing[j]] = name_hint or 'model'
    return current_result, sources


def infer_ranges_from_specs_or_data(feature_names, df):
    ranges = {}
    if os.path.exists(SPEC_XLSX):
        try:
            spec_df = pd.read_excel(SPEC_XLSX)
            param_col = None
            min_col = None
            max_col = None
            for c in spec_df.columns:
                lc = c.lower()
                if 'param' in lc or 'parameter' in lc or 'name' in lc:
                    param_col = c
                if lc in ('min', 'minimum'):
                    min_col = c
                if lc in ('max', 'maximum'):
                    max_col = c
            if param_col and min_col and max_col:
                for _, row in spec_df.iterrows():
                    p = row[param_col]
                    if p in feature_names:
                        try:
                            ranges[p] = (float(row[min_col]), float(row[max_col]))
                        except Exception:
                            pass
        except Exception:
            pass

    for fname in feature_names:
        if fname not in ranges:
            col = df[fname].dropna()
            if len(col) > 10:
                lo = float(col.quantile(0.01))
                hi = float(col.quantile(0.99))
            else:
                lo = float(col.min())
                hi = float(col.max())
            padding = max(abs(0.05 * lo), 1e-6)
            ranges[fname] = (lo - padding, hi + padding)
    return ranges

# ------------------ App logic ------------------

def main():
    df = load_dataset()
    feature_names = df.columns.tolist()

    st.sidebar.header("Predictor & Models")
    st.sidebar.write("Files expected in repo root: diesel_properties_clean.xlsx, scaler.joblib, pls_model.joblib, rf_model.zip")
    predictor = st.sidebar.selectbox("Predictor", options=["Auto (imputer + models)", "Imputer only", "PLS only", "RF only"]) 
    st.sidebar.markdown("---")

    need_refit = True
    if 'imputer' in st.session_state:
        existing = st.session_state['imputer']
        n_in = getattr(existing, 'n_features_in_', None)
        if n_in == len(feature_names):
            need_refit = False
    if need_refit:
        with st.spinner('Fitting imputer on dataset...'):
            st.session_state['imputer'] = build_imputer(df)
    imputer = st.session_state['imputer']

    scaler = None
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            scaler = None

    pls_model = None
    if os.path.exists(PLS_PATH):
        try:
            pls_model = joblib.load(PLS_PATH)
        except Exception as e:
            st.sidebar.warning(f"Couldn't load PLS model: {e}")

    rf_models = extract_models_from_zip(RF_ZIP_PATH)
    ranges = infer_ranges_from_specs_or_data(feature_names, df)

    # ---------------- INPUT SECTIONS WITH HEADINGS ----------------
    st.subheader("Inputs")
    st.write("Provide any two parameter values. Leave others blank.")

    user_inputs = {}
    cols = st.columns(3)

    for i, fname in enumerate(feature_names):
        with cols[i % 3]:

            # NEW: Heading above each input
            st.markdown(f"### {fname}")

            inp = st.text_input(
                fname,
                key=f"inp_{fname}",
                placeholder=f"Enter {fname} (e.g. {df[fname].median():.3g})"
            )

            lo, hi = ranges.get(fname, (None, None))
            if lo is not None and hi is not None:
                st.markdown(
                    f"<div class='small-muted'>Range: {lo:.6g} — {hi:.6g}</div>",
                    unsafe_allow_html=True
                )

            if inp is not None and inp.strip() != "":
                try:
                    v = float(inp)
                    user_inputs[fname] = v
                except Exception:
                    st.info(f"{fname}: Unable to parse '{inp}' as a number.")

    st.markdown("---")
    predict_btn = st.button("Predict", key="predict")

    if predict_btn:
        if len(user_inputs) < 2:
            st.error("Please enter at least two numeric parameters.")
        else:
            X_row = create_input_row(feature_names, user_inputs)
            X_imputed = imputer.transform(X_row)
            result = dict(zip(feature_names, X_imputed.ravel().tolist()))
            sources = {f: ('user' if f in user_inputs else 'imputer') for f in feature_names}

            X_for_models = X_imputed.copy()
            if scaler is not None:
                try:
                    X_for_models = scaler.transform(X_imputed)
                except Exception:
                    X_for_models = X_imputed.copy()

            if predictor in ("RF only", "Auto (imputer + models)") and rf_models:
                for name, model in rf_models.items():
                    result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                    sources.update(new_sources)

            if predictor in ("PLS only", "Auto (imputer + models)") and pls_model is not None:
                try:
                    result, new_sources = apply_model_predictions(pls_model, X_for_models, feature_names, result, user_inputs, name_hint=PLS_PATH)
                    sources.update(new_sources)
                except Exception:
                    pass

            out_df = pd.DataFrame([result]).T
            out_df.columns = ["Value"]
            out_df["Source"] = pd.Series(sources)
            st.subheader("Predicted / Completed Properties")
            st.dataframe(out_df.round(6))

            csv = pd.DataFrame([result]).to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="predicted_fuel_properties.csv", mime="text/csv")

            st.write("**Inputs provided:**")
            st.json(user_inputs)
            st.caption("Source legend: 'user' = provided by you; 'imputer' = filled by IterativeImputer; others = model filename used.")

if __name__ == '__main__':
    main()