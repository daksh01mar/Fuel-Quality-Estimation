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
    # drop obvious non-feature columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for c in ["ID", "Sample", "SampleID", "index", "Unnamed: 0"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    # keep numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove any accidental 'LABEL' column (case-insensitive)
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
    """
    Try to apply a model's predict output to fill missing features.
    Returns (updated_result_dict, sources_dict)
    """
    sources = {}
    try:
        ypred = model.predict(X_for_models)
    except Exception as e:
        st.warning(f"Model {name_hint or str(model)} failed to predict: {e}")
        return current_result, sources
    ypred = np.asarray(ypred)
    n_features = len(feature_names)
    # Many possible shapes -> decode sensibly
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
        # Fallback: flatten and assign to missing features
        missing = [f for f in feature_names if f not in user_inputs]
        flat = ypred.ravel()
        for j in range(min(len(flat), len(missing))):
            current_result[missing[j]] = float(flat[j])
            sources[missing[j]] = name_hint or 'model'
    return current_result, sources


def infer_ranges_from_specs_or_data(feature_names, df):
    ranges = {}
    # try to read SPEC_XLSX if present and has min/max columns
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
    # fallback to dataset percentiles if not found
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


def safe_apply_pls(pls_model, X_for_models, feature_names, base_result, user_inputs, imputer=None):
    """
    Attempt to apply a PLS model while handling mismatched feature-counts safely.
    Returns updated_result_dict and updated_sources_dict.
    """
    sources = {}
    result_updates = {}

    req = getattr(pls_model, "n_features_in_", None)
    fname_in = getattr(pls_model, "feature_names_in_", None)
    cur = X_for_models.shape[1]

    # Case A: exact match -> apply directly
    if req is not None and req == cur:
        try:
            result_updates, sources = apply_model_predictions(pls_model, X_for_models, feature_names, base_result.copy(), user_inputs, name_hint=os.path.basename(PLS_PATH))
            return result_updates, sources
        except Exception as e:
            st.warning(f"PLS predict failed: {e}")
            return {}, {}

    # Case B: if model has feature_names_in_, attempt name-based mapping
    if fname_in is not None:
        fname_in = list(fname_in)
        X_mapped = np.full((1, len(fname_in)), np.nan, dtype=float)
        cur_map = {n: i for i, n in enumerate(feature_names)}
        for j, req_name in enumerate(fname_in):
            if req_name in cur_map:
                X_mapped[0, j] = X_for_models[0, cur_map[req_name]]
        # if any NaNs remain, try to impute (if imputer provided)
        if np.isnan(X_mapped).any():
            if imputer is not None:
                try:
                    X_mapped = imputer.transform(X_mapped)
                except Exception:
                    st.warning("Couldn't impute missing PLS-mapped features; some values remain NaN.")
            else:
                st.warning("PLS expects features not present in the app and no imputer available to fill them.")
        # attempt prediction
        try:
            ypred = pls_model.predict(X_mapped)
        except Exception as e:
            st.warning(f"PLS predict failed on mapped input: {e}")
            return {}, {}
        # wrap ypred into a fake model for decoding using apply_model_predictions
        class _FakeModel:
            def __init__(self, y): self._y = y
            def predict(self, X): return self._y

        fake = _FakeModel(ypred)
        try:
            result_updates, sources = apply_model_predictions(fake, X_for_models, feature_names, base_result.copy(), user_inputs, name_hint=os.path.basename(PLS_PATH))
            # Indicate source as PLS where applicable
            for k in list(sources.keys()):
                sources[k] = os.path.basename(PLS_PATH)
            return result_updates, sources
        except Exception:
            st.warning("Couldn't decode PLS outputs into named features after mapping.")
            return {}, {}

    # Case C: attempt padding if model expects more features (risky)
    if req is not None and req > cur:
        st.warning(
            f"PLS model expects {req} features but app provides {cur}. Attempting to pad missing inputs with zeros (this is risky)."
        )
        X_pad = np.zeros((1, req), dtype=float)
        X_pad[0, :cur] = X_for_models[0]
        try:
            ypred = pls_model.predict(X_pad)
        except Exception as e:
            st.warning(f"PLS predict failed after padding: {e}")
            return {}, {}
        class _FakeModel:
            def __init__(self, y): self._y = y
            def predict(self, X): return self._y
        fake = _FakeModel(ypred)
        try:
            result_updates, sources = apply_model_predictions(fake, X_for_models, feature_names, base_result.copy(), user_inputs, name_hint=os.path.basename(PLS_PATH))
            for k in list(sources.keys()):
                sources[k] = os.path.basename(PLS_PATH)
            return result_updates, sources
        except Exception:
            st.warning("Couldn't decode PLS outputs into named features after padding.")
            return {}, {}

    # Otherwise incompatible
    st.info("PLS model appears incompatible with current feature set and was skipped.")
    return {}, {}


# ------------------ App logic ------------------


def main():
    df = load_dataset()
    feature_names = df.columns.tolist()

    st.sidebar.header("Predictor & Models")
    st.sidebar.write("Files expected in repo root: diesel_properties_clean.xlsx, scaler.joblib (optional), pls_model.joblib (optional), rf_model.zip (optional)")
    predictor = st.sidebar.selectbox("Predictor", options=["Auto (imputer + models)", "Imputer only", "PLS only", "RF only"])
    st.sidebar.markdown("---")
    st.sidebar.write("You can upload different model artifacts directly to the repository root and redeploy.")

    # cache imputer to speed up
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

    # load optional models
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

    # compute ranges to display under inputs
    ranges = infer_ranges_from_specs_or_data(feature_names, df)

    # Build a clean grid of inputs (no extra checkboxes)
    st.subheader("Inputs")
    st.write("Provide any two parameter values. Leave others blank.")

    user_inputs = {}
    cols = st.columns(3)
    for i, fname in enumerate(feature_names):
        with cols[i % 3]:
            inp = st.text_input(
                "",
                key=f"inp_{fname}",
                placeholder=f"Enter {fname} (e.g. {df[fname].median():.3g})",
                label_visibility="collapsed"
            )
            lo, hi = ranges.get(fname, (None, None))
            if lo is not None and hi is not None:
                st.markdown(f"<div class='small-muted'>Range: {lo:.6g} — {hi:.6g}</div>", unsafe_allow_html=True)
            if inp is not None and inp.strip() != "":
                try:
                    v = float(inp)
                    user_inputs[fname] = v
                except Exception:
                    st.info(f"{fname}: Unable to parse '{inp}' as a number. Please enter a numeric value.")
    st.markdown("---")
    predict_btn = st.button("Predict", key="predict")

    if predict_btn:
        if len(user_inputs) < 2:
            st.error("Please enter at least two numeric parameters.")
        else:
            X_row = create_input_row(feature_names, user_inputs)
            X_imputed = imputer.transform(X_row)
            # Base result: imputer-filled values (so all features have values)
            result = dict(zip(feature_names, X_imputed.ravel().tolist()))
            sources = {f: ('user' if f in user_inputs else 'imputer') for f in feature_names}

            X_for_models = X_imputed.copy()
            if scaler is not None:
                try:
                    X_for_models = scaler.transform(X_imputed)
                except Exception as e:
                    st.warning(f"Scaler transform failed: {e}. Proceeding with unscaled features.")

            # RF models
            if predictor in ("RF only", "Auto (imputer + models)") and rf_models:
                if len(rf_models) == 1:
                    name, model = next(iter(rf_models.items()))
                    result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                    sources.update(new_sources)
                else:
                    for name, model in rf_models.items():
                        result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                        sources.update(new_sources)

            # PLS model (safe)
            if predictor in ("PLS only", "Auto (imputer + models)") and pls_model is not None:
                # Quick inspection
                req = getattr(pls_model, "n_features_in_", None)
                if req is not None and req != X_for_models.shape[1] and not getattr(pls_model, "feature_names_in_", None):
                    st.warning(f"PLS model expects {req} features but app has {X_for_models.shape[1]}. The app will attempt safe mapping/padding; retraining is recommended for correct results.")
                updates, pls_sources = safe_apply_pls(pls_model, X_for_models, feature_names, result, user_inputs, imputer=imputer)
                if updates:
                    # merge updates into result and sources
                    for k, v in updates.items():
                        result[k] = v
                    sources.update(pls_sources)

            # display
            out_df = pd.DataFrame([result]).T
            out_df.columns = ["Value"]
            out_df["Source"] = pd.Series(sources)
            out_df_display = out_df.round(6)
            st.subheader("Predicted / Completed Properties")
            st.dataframe(out_df_display)

            csv = pd.DataFrame([result]).to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="predicted_fuel_properties.csv", mime="text/csv")

            st.write("**Inputs provided:**")
            st.json(user_inputs)
            st.caption("Source legend: 'user' = provided by you; 'imputer' = filled by IterativeImputer; others = model filename used.")


if __name__ == '__main__':
    main()
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
    # drop obvious non-feature columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for c in ["ID", "Sample", "SampleID", "index", "Unnamed: 0"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    # keep numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove any accidental 'LABEL' column (case-insensitive)
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
    sources = {}
    try:
        ypred = model.predict(X_for_models)
    except Exception as e:
        st.warning(f"Model {name_hint or str(model)} failed to predict: {e}")
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
    # try to read SPEC_XLSX if present and has min/max columns
    if os.path.exists(SPEC_XLSX):
        try:
            spec_df = pd.read_excel(SPEC_XLSX)
            # try to find columns that match feature names and min/max
            for fname in feature_names:
                if fname in spec_df.columns:
                    # if column contains a range string like '820-845' attempt parse, else skip
                    # alternatively check for spec_df rows with 'min'/'max' columns
                    pass
            # look for 'Parameter','Min','Max' style
            param_col = None
            min_col = None
            max_col = None
            for c in spec_df.columns:
                lc = c.lower()
                if 'param' in lc or 'parameter' in lc or 'name' in lc:
                    param_col = c
                if lc in ('min','minimum'):
                    min_col = c
                if lc in ('max','maximum'):
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
    # fallback to dataset percentiles if not found
    for fname in feature_names:
        if fname not in ranges:
            col = df[fname].dropna()
            if len(col) > 10:
                lo = float(col.quantile(0.01))
                hi = float(col.quantile(0.99))
            else:
                lo = float(col.min())
                hi = float(col.max())
            # small padding
            padding = max(abs(0.05 * lo), 1e-6)
            ranges[fname] = (lo - padding, hi + padding)
    return ranges

# ------------------ App logic ------------------

def main():
    df = load_dataset()
    feature_names = df.columns.tolist()

    st.sidebar.header("Predictor & Models")
    st.sidebar.write("Files expected in repo root: diesel_properties_clean.xlsx, scaler.joblib (optional), pls_model.joblib (optional), rf_model.zip (optional)")
    predictor = st.sidebar.selectbox("Predictor", options=["Auto (imputer + models)", "Imputer only", "PLS only", "RF only"]) 
    st.sidebar.markdown("---")
    st.sidebar.write("You can upload different model artifacts directly to the repository root and redeploy.")

    # cache imputer to speed up
    # If the dataset features change (e.g. LABEL column removed) we must refit the imputer.
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

    # load optional models
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

    # compute ranges to display under inputs
    ranges = infer_ranges_from_specs_or_data(feature_names, df)

    # Build a clean grid of inputs (no extra checkboxes)
    st.subheader("Inputs")
    st.write("Provide any two parameter values. Leave others blank.")

    user_inputs = {}
    cols = st.columns(3)
    for i, fname in enumerate(feature_names):
        with cols[i % 3]:
            inp = st.text_input(
                "",
                key=f"inp_{fname}",
                placeholder=f"Enter {fname} (e.g. {df[fname].median():.3g})",
                label_visibility="collapsed"
            )
            lo, hi = ranges.get(fname, (None, None))
            if lo is not None and hi is not None:
                st.markdown(f"<div class='small-muted'>Range: {lo:.6g} — {hi:.6g}</div>", unsafe_allow_html=True)
            # try parse
            if inp is not None and inp.strip() != "":
                try:
                    v = float(inp)
                    user_inputs[fname] = v
                except Exception:
                    st.info(f"{fname}: Unable to parse '{inp}' as a number. Please enter a numeric value.")
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
                except Exception as e:
                    st.warning(f"Scaler transform failed: {e}. Proceeding with unscaled features.")

            # RF
            if predictor in ("RF only", "Auto (imputer + models)") and rf_models:
                if len(rf_models) == 1:
                    name, model = next(iter(rf_models.items()))
                    result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                    sources.update(new_sources)
                else:
                    for name, model in rf_models.items():
                        result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                        sources.update(new_sources)
            # PLS
            if predictor in ("PLS only", "Auto (imputer + models)") and pls_model is not None:
                result, new_sources = apply_model_predictions(pls_model, X_for_models, feature_names, result, user_inputs, name_hint=os.path.basename(PLS_PATH))
                sources.update(new_sources)

            # display
            out_df = pd.DataFrame([result]).T
            out_df.columns = ["Value"]
            out_df["Source"] = pd.Series(sources)
            out_df_display = out_df.round(6)
            st.subheader("Predicted / Completed Properties")
            st.dataframe(out_df_display)

            csv = pd.DataFrame([result]).to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="predicted_fuel_properties.csv", mime="text/csv")

            st.write("**Inputs provided:**")
            st.json(user_inputs)
            st.caption("Source legend: 'user' = provided by you; 'imputer' = filled by IterativeImputer; others = model filename used.")

if __name__ == '__main__':
    main()
