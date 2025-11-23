import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import tempfile
import os
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ------------------ Configuration (uploaded files) ------------------
DATA_XLSX = "/mnt/data/diesel_properties_clean.xlsx"
SPEC_XLSX = "/mnt/data/diesel_spec.xlsx"  # optional
SCALER_PATH = "/mnt/data/scaler.joblib"
PLS_PATH = "/mnt/data/pls_model.joblib"
RF_ZIP_PATH = "/mnt/data/rf_model.zip"
# ------------------------------------------------------------------

st.set_page_config(page_title="Fuel Parameter Predictor", layout="wide")

# Minimal dark theme tweaks
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .stButton>button { background-color: #1f2937; color: #e6edf3; }
    .stTextInput>div>div>input { background-color: #0b1220; color: #e6edf3; }
    .stSelectbox>div>div { background-color: #0b1220; color: #e6edf3; }
    .stFileUploader>div { background-color: #0b1220; color: #e6edf3; }
    table { color: #e6edf3; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fuel parameter predictor (Streamlit)")
st.caption("Supply any two numeric parameters and choose a predictor: Imputer (default), PLS, RF, or Auto.")

# ------------------ Helper functions ------------------

def load_dataset(path=DATA_XLSX):
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}")
        st.stop()
    df = pd.read_excel(path)
    # drop obvious non-feature columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for c in ["ID", "Sample", "SampleID", "index", "Unnamed: 0"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    # keep numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
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
                except Exception as e:
                    st.warning(f"Couldn't load {p.name} from zip: {e}")
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
    """Try to apply model predictions to feature dict. Returns updated result and sources."""
    sources = {}
    try:
        ypred = model.predict(X_for_models)
    except Exception as e:
        st.warning(f"Model {name_hint or str(model)} failed to predict: {e}")
        return current_result, sources
    ypred = np.asarray(ypred)
    n_features = len(feature_names)
    # many shapes handled permissively
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
        # fallback: try to assign first values to first missing
        missing = [f for f in feature_names if f not in user_inputs]
        flat = ypred.ravel()
        for j in range(min(len(flat), len(missing))):
            current_result[missing[j]] = float(flat[j])
            sources[missing[j]] = name_hint or 'model'
    return current_result, sources


# ------------------ App logic ------------------

def main():
    df = load_dataset()
    feature_names = df.columns.tolist()
    st.sidebar.header("Files & models (fixed)")
    st.sidebar.write("Dataset:")
    st.sidebar.text(DATA_XLSX)
    st.sidebar.write("PLS:")
    st.sidebar.text(PLS_PATH if os.path.exists(PLS_PATH) else "(not found)")
    st.sidebar.write("Scaler:")
    st.sidebar.text(SCALER_PATH if os.path.exists(SCALER_PATH) else "(not found)")
    st.sidebar.write("RF zip:")
    st.sidebar.text(RF_ZIP_PATH if os.path.exists(RF_ZIP_PATH) else "(not found)")

    # build imputer (could be slow; do it once and cache)
    if 'imputer' not in st.session_state:
        with st.spinner('Fitting imputer on dataset...'):
            st.session_state['imputer'] = build_imputer(df)
    imputer = st.session_state['imputer']

    # load optional artifacts
    scaler = None
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            st.sidebar.warning(f"Couldn't load scaler: {e}")

    pls_model = None
    if os.path.exists(PLS_PATH):
        try:
            pls_model = joblib.load(PLS_PATH)
        except Exception as e:
            st.sidebar.warning(f"Couldn't load PLS model: {e}")

    rf_models = extract_models_from_zip(RF_ZIP_PATH)

    st.sidebar.header("Predictor options")
    predictor = st.sidebar.selectbox("Predictor", options=["Auto (imputer + models)", "Imputer only", "PLS only", "RF models only"]) 
    show_raw = st.sidebar.checkbox("Show raw imputed vector", value=False)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input parameters")
        st.write("Provide any TWO numeric parameters. Leave others blank.")
        user_inputs = {}
        # Provide input fields for all features but recommend only two
        for fname in feature_names:
            val = st.number_input(label=fname, key=f"inp_{fname}", format="%.6g")
            # number_input returns 0.0 default; we want blank semantics. To allow blank, use text_input and try parse.
            # But to keep simple, we'll interpret 0.0 as a valid entry if user explicitly checks a box.
            # Instead provide a checkbox to indicate if provided
            provided = st.checkbox(f"Provide {fname}", key=f"chk_{fname}")
            if provided:
                txt = st.text_input(f"Value for {fname}", key=f"txt_{fname}")
                try:
                    v = float(txt)
                    user_inputs[fname] = v
                except Exception:
                    st.warning(f"Invalid numeric for {fname}. Leave blank or enter a number.")

        if len(user_inputs) < 2:
            st.info("Please provide at least two parameters (tick Provide and enter value).")

        predict_btn = st.button("Predict")

    with col2:
        st.subheader("Results")
        result_area = st.empty()

    if predict_btn:
        if len(user_inputs) < 2:
            st.error("You must provide at least two parameters.")
        else:
            # Prepare input
            X_row = create_input_row(feature_names, user_inputs)
            X_imputed = imputer.transform(X_row)
            # base result from imputer
            result = dict(zip(feature_names, X_imputed.ravel().tolist()))
            sources = {f: ('user' if f in user_inputs else 'imputer') for f in feature_names}

            # Optionally show raw imputed vector
            if show_raw:
                imputed_df = pd.DataFrame([result])
                st.write("Imputer result:")
                st.dataframe(imputed_df)

            # prepare scaled vector for models
            X_for_models = X_imputed.copy()
            if scaler is not None:
                try:
                    X_for_models = scaler.transform(X_imputed)
                except Exception as e:
                    st.warning(f"Scaler transform failed: {e}. Proceeding with unscaled features.")

            # Apply models based on predictor choice
            if predictor in ("RF models only", "Auto (imputer + models)") and rf_models:
                # if single model, apply; if many, try per-file names
                if len(rf_models) == 1:
                    name, model = next(iter(rf_models.items()))
                    result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                    sources.update(new_sources)
                else:
                    for name, model in rf_models.items():
                        result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                        sources.update(new_sources)
            if predictor in ("PLS only", "Auto (imputer + models)") and pls_model is not None:
                result, new_sources = apply_model_predictions(pls_model, X_for_models, feature_names, result, user_inputs, name_hint=os.path.basename(PLS_PATH))
                sources.update(new_sources)

            # Prepare result table
            out_df = pd.DataFrame([result])
            # round for display
            out_df_display = out_df.round(6)
            # attach source row
            src_row = pd.DataFrame([sources])
            display_df = out_df_display.T
            display_df.columns = ["Value"]
            display_df["Source"] = src_row.T

            result_area.dataframe(display_df)

            # allow CSV download
            csv = out_df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="predicted_fuel_properties.csv", mime="text/csv")

            # show which inputs were used
            st.write("Inputs provided:")
            st.json(user_inputs)

            # small note
            st.caption("Source legend: 'user' = provided by you; 'imputer' = IterativeImputer; others = model filename used.")


if __name__ == '__main__':
    main()
