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

st.set_page_config(page_title="Fuel Parameter Predictor + Quality", layout="wide")

# ------------------ Quality scoring config ------------------
SPECS = {
    "CN":     {"type": "ge",    "min": 51.0,              "label": "Cetane Number"},
    "D4052":  {"type": "range", "min": 820.0, "max": 845.0, "label": "Density @15°C (kg/m³)"},
    "VISC":   {"type": "range", "min": 2.0,   "max": 4.5,   "label": "Viscosity @40°C (mm²/s)"},
    "FLASH":  {"type": "ge",    "min": 66.0,              "label": "Flash Point (°C)"},
    "BP50":   {"type": "range", "min": 245.0, "max": 350.0, "label": "Distillation T50 (°C)"},
    "FREEZE": {"type": "le",    "max": -20.0,             "label": "Freeze Point (°C)"},
    "TOTAL":  {"type": "le",    "max": 50.0,              "label": "Total Sulfur (ppm)"},
}
TOL = {
    "CN": 10.0,
    "D4052": 10.0,
    "VISC": 1.0,
    "FLASH": 10.0,
    "BP50": 25.0,
    "FREEZE": 10.0,
    "TOTAL": 20.0,
}
WEIGHTS = {
    "CN": 0.20,
    "TOTAL": 0.20,
    "VISC": 0.15,
    "D4052": 0.15,
    "FLASH": 0.10,
    "BP50": 0.10,
    "FREEZE": 0.10,
}

# ------------------ Scoring functions ------------------
def score_ge(val, min_, tol):
    if val >= min_:
        return 1.0
    return float(np.clip((val - (min_ - tol)) / tol, 0.0, 1.0))

def score_le(val, max_, tol):
    if val <= max_:
        return 1.0
    return float(np.clip(((max_ + tol) - val) / tol, 0.0, 1.0))

def score_range(val, low, high, tol):
    if low <= val <= high:
        return 1.0
    if val < low:
        dist = low - val
    else:
        dist = val - high
    return float(np.clip(1.0 - dist / tol, 0.0, 1.0))

def compute_property_score(prop, val):
    spec = SPECS[prop]
    tol = TOL[prop]
    if spec["type"] == "ge":
        s = score_ge(val, spec["min"], tol)
        passed = val >= spec["min"]
    elif spec["type"] == "le":
        s = score_le(val, spec["max"], tol)
        passed = val <= spec["max"]
    elif spec["type"] == "range":
        s = score_range(val, spec["min"], spec["max"], tol)
        passed = (spec["min"] <= val <= spec["max"])
    else:
        s, passed = np.nan, False
    return s, passed

def classify_quality(score_0_100):
    if score_0_100 >= 85:
        return "Excellent"
    elif score_0_100 >= 70:
        return "Good"
    elif score_0_100 >= 50:
        return "Marginal"
    else:
        return "Reject"

# ------------------ Helper functions for prediction app ------------------

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
    # remove 'label' column if present
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

# ------------------ UI: render quality ------------------
def render_quality_from_values(vals):
    prop_rows = []
    weighted_sum = 0.0
    weight_total = 0.0
    all_pass = True
    for p, v in vals.items():
        s_01, passed = compute_property_score(p, v)
        s_100 = round(100.0 * s_01, 2)
        w = WEIGHTS[p]
        if not np.isnan(s_01):
            weighted_sum += s_01 * w
            weight_total += w
        all_pass = all_pass and passed
        spec = SPECS[p]
        if spec["type"] == "ge":
            spec_txt = f"≥ {spec['min']}"
        elif spec["type"] == "le":
            spec_txt = f"≤ {spec['max']}"
        else:
            spec_txt = f"{spec['min']} – {spec['max']}"
        prop_rows.append({
            "Property": f"{p} ({spec['label']})",
            "Value": round(v, 3),
            "Spec limit": spec_txt,
            "Pass (strict)": "YES" if passed else "NO",
            "Score (0–100)": s_100,
        })
    if weight_total > 0:
        adv_score = round(100.0 * weighted_sum / weight_total, 2)
    else:
        adv_score = np.nan
    quality_class = classify_quality(adv_score)
    st.markdown("### Results")
    st.metric("Advanced Quality Score", f"{adv_score} / 100", help="Weighted index combining all properties.")
    st.metric("Quality Class", quality_class)
    if all_pass:
        st.success("All properties meet the strict specification limits (PASS).")
    else:
        st.warning("One or more properties fail the strict spec. Check the table below for details.")
    st.markdown("#### Per-property details")
    st.dataframe(prop_rows, use_container_width=True)
    st.markdown(
        "> Note: The **advanced score** is smoother than simple Pass/Fail. "
        "Even if a property slightly violates the limit, it does not drop the quality to 0 instantly; "
        "instead, the score gradually decreases based on how far it is from the spec."
    )

# ------------------ App UI ------------------

def main():
    st.sidebar.title("App mode")
    mode = st.sidebar.selectbox("Choose mode", ["Predict parameters", "Estimate quality (manual)", "Estimate quality (from prediction)"])

    # Load dataset and models only if prediction mode selected
    if mode == "Predict parameters":
        df = load_dataset()
        feature_names = df.columns.tolist()
        st.header("Predict missing fuel parameters")
        st.write("Provide any two parameters below. The app will impute missing properties and optionally apply models if available.")
        # prepare imputer
        if 'imputer' not in st.session_state:
            with st.spinner('Fitting imputer on dataset...'):
                st.session_state['imputer'] = build_imputer(df)
        else:
            # ensure imputer matches features
            existing = st.session_state['imputer']
            n_in = getattr(existing, 'n_features_in_', None)
            if n_in != len(feature_names):
                with st.spinner('Refitting imputer (feature set changed)...'):
                    st.session_state['imputer'] = build_imputer(df)
        imputer = st.session_state['imputer']
        scaler = None
        if os.path.exists(SCALER_PATH):
            try:
                scaler = joblib.load(SCALER_PATH)
            except Exception:
                st.sidebar.warning('Could not load scaler')
        pls_model = None
        if os.path.exists(PLS_PATH):
            try:
                pls_model = joblib.load(PLS_PATH)
            except Exception:
                st.sidebar.warning('Could not load PLS')
        rf_models = extract_models_from_zip(RF_ZIP_PATH)

        # ranges
        ranges = infer_ranges_from_specs_or_data(feature_names, df)

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
        if st.button("Predict"):
            if len(user_inputs) < 2:
                st.error("Please input at least two parameters.")
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
                        st.warning('Scaler transform failed')
                if rf_models:
                    if len(rf_models) == 1:
                        name, model = next(iter(rf_models.items()))
                        result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                        sources.update(new_sources)
                    else:
                        for name, model in rf_models.items():
                            result, new_sources = apply_model_predictions(model, X_for_models, feature_names, result, user_inputs, name_hint=name)
                            sources.update(new_sources)
                if pls_model is not None:
                    result, new_sources = apply_model_predictions(pls_model, X_for_models, feature_names, result, user_inputs, name_hint=os.path.basename(PLS_PATH))
                    sources.update(new_sources)

                st.subheader("Predicted / Completed Properties")
                out_df = pd.DataFrame([result]).T
                out_df.columns = ["Value"]
                out_df["Source"] = pd.Series(sources)
                st.dataframe(out_df.round(6))

                # Offer to compute quality from predicted values
                if st.button("Compute quality from predicted values"):
                    # map to required keys for quality (ensure names match SPECS keys)
                    vals = {
                        "CN": result.get('CN'),
                        "D4052": result.get('D4052'),
                        "VISC": result.get('VISC'),
                        "FLASH": result.get('FLASH'),
                        "BP50": result.get('BP50'),
                        "FREEZE": result.get('FREEZE'),
                        "TOTAL": result.get('TOTAL'),
                    }
                    render_quality_from_values(vals)

    elif mode == "Estimate quality (manual)":
        st.header("Estimate fuel quality (manual inputs)")
        st.write("Enter measured or predicted properties to compute the advanced quality score.")
        col1, col2 = st.columns(2)
        with col1:
            cn = st.number_input("Cetane Number (CN)", min_value=0.0, max_value=80.0, value=50.0, step=0.1)
            dens = st.number_input("Density D4052 @15°C (kg/m³)", min_value=700.0, max_value=900.0, value=835.0, step=0.5)
            visc = st.number_input("Viscosity @40°C (mm²/s)", min_value=0.0, max_value=10.0, value=3.0, step=0.05)
            flash = st.number_input("Flash Point (°C)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        with col2:
            bp50 = st.number_input("Distillation T50 (°C)", min_value=150.0, max_value=400.0, value=270.0, step=1.0)
            freeze = st.number_input("Freeze Point (°C)", min_value=-60.0, max_value=20.0, value=-15.0, step=0.5)
            sulfur = st.number_input("Total Sulfur (ppm)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
        if st.button("Estimate Fuel Quality"):
            vals = {
                "CN": cn,
                "D4052": dens,
                "VISC": visc,
                "FLASH": flash,
                "BP50": bp50,
                "FREEZE": freeze,
                "TOTAL": sulfur,
            }
            render_quality_from_values(vals)

    else:  # Estimate quality (from prediction)
        st.header("Estimate quality from prediction")
        st.write("First run 'Predict parameters' mode to get completed properties, then use this mode to load them and compute quality.")
        st.info("This mode reads a CSV named 'predicted_fuel_properties.csv' from the repo root if present.")
        if os.path.exists('predicted_fuel_properties.csv'):
            try:
                pred_df = pd.read_csv('predicted_fuel_properties.csv')
                # expect single-row
                if pred_df.shape[0] >= 1:
                    row = pred_df.iloc(0).to_dict()
                    vals = {
                        "CN": row.get('CN'),
                        "D4052": row.get('D4052'),
                        "VISC": row.get('VISC'),
                        "FLASH": row.get('FLASH'),
                        "BP50": row.get('BP50'),
                        "FREEZE": row.get('FREEZE'),
                        "TOTAL": row.get('TOTAL'),
                    }
                    st.write("Loaded predicted properties from predicted_fuel_properties.csv")
                    render_quality_from_values(vals)
                else:
                    st.info('predicted_fuel_properties.csv is empty or invalid format')
            except Exception as e:
                st.error(f"Failed to read predicted_fuel_properties.csv: {e}")
        else:
            st.info("No predicted_fuel_properties.csv found in repo root. Use 'Predict parameters' mode and download the CSV, then upload it to the repo if you want to reuse it here.")

if __name__ == '__main__':
    main()
