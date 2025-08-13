# app.py
# Purpose:
# Streamlit web application for phishing URL analysis using:
# - A trained PyCaret classification model for predictive scoring.
# - A trained PyCaret clustering model for threat actor attribution.
# - Generative AI (Gemini, OpenAI, or Grok) for prescriptive response planning.
# - Risk feature visualization for analyst context.

import os
import time
import pandas as pd
import streamlit as st
from pycaret.classification import load_model as load_clf_model, predict_model as predict_clf
from pycaret.clustering   import load_model as load_clu_model, predict_model as predict_clu
from genai_prescriptions  import generate_prescription

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="GenAI-Powered Phishing SOAR",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Paths (one place to change if layout moves)
# -----------------------------
CLF_PATH = 'models/phishing_url_detector'
CLU_PATH = 'models/threat_actor_profiler'
FEATURE_PLOT_PATH = 'models/feature_importance.png'
CLUSTER_CSV = 'data/phishing_with_clusters.csv'

# -----------------------------
# Threat Actor Mapping (cluster -> profile)
# Update after inspecting data/phishing_with_clusters.csv
# -----------------------------
CLUSTER_TO_PROFILE = {
    0: "Organized Cybercrime",
    1: "State-Sponsored",
    2: "Hacktivist",
}
PROFILE_DESCRIPTIONS = {
    "State-Sponsored": (
        "Highly resourced, stealthy campaigns. Often use valid SSL and subtle obfuscation "
        "(e.g., hyphenated domains), long-lived infrastructure, and targeted spear-phishing "
        "aligned to geopolitical objectives."
    ),
    "Organized Cybercrime": (
        "Profit-driven, high-volume operations. Noisy patterns like URL shorteners, raw IPs, "
        "abnormal paths, and aggressive redirect chains; goals include credential theft, "
        "ransomware, or fraud at scale."
    ),
    "Hacktivist": (
        "Cause-driven and opportunistic. Mixed tradecraft and activist/issue keywords; activity "
        "may spike around news cycles, leaks, or protests with defacement or doxxing motives."
    ),
}

# -----------------------------
# Load models & feature plot (cached)
# -----------------------------
@st.cache_resource
def load_assets():
    """Loads trained models (classifier + clustering) and feature importance plot."""
    clf_model = load_clf_model(CLF_PATH) if os.path.exists(CLF_PATH + '.pkl') else None
    clu_model = load_clu_model(CLU_PATH) if os.path.exists(CLU_PATH + '.pkl') else None
    plot_path = FEATURE_PLOT_PATH if os.path.exists(FEATURE_PLOT_PATH) else None
    return clf_model, clu_model, plot_path

model, cluster_model, feature_plot = load_assets()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🔬 URL Feature Input")
    st.write("Describe the characteristics of a suspicious URL below.")

    form_values = {
        'url_length':  st.select_slider("URL Length", options=['Short', 'Normal', 'Long'], value='Long'),
        'ssl_state':   st.select_slider("SSL Certificate Status", options=['Trusted', 'Suspicious', 'None'], value='Suspicious'),
        'sub_domain':  st.select_slider("Sub-domain Complexity", options=['None', 'One', 'Many'], value='One'),
        'prefix_suffix':         st.checkbox("URL has a Prefix/Suffix (e.g., '-')", value=True),
        'has_ip':                st.checkbox("URL uses an IP Address", value=False),
        'short_service':         st.checkbox("Is it a shortened URL", value=False),
        'at_symbol':             st.checkbox("URL contains '@' symbol", value=False),
        'abnormal_url':          st.checkbox("Is it an abnormal URL", value=True),
        'has_political_keyword': st.checkbox("Contains political/activist keyword", value=False),
    }

    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button("💥 Analyze & Initiate Response", use_container_width=True, type="primary")

    # --- Utilities ---
    st.divider()
    show_debug = st.checkbox("Show clustering debug", value=False)  # optional
    if st.button("↻ Reload models (clear cache)", use_container_width=True):
        load_assets.clear()
        st.success("Cache cleared. Reloading…")
        st.experimental_rerun()

# -----------------------------
# Guardrail: classifier required
# -----------------------------
if not model:
    st.error(
        "Classifier model not found. Run training to generate artifacts:\n\n"
        "• models/phishing_url_detector.pkl\n"
        "• models/feature_importance.png\n\n"
        "Use `python train_model.py` or `make train`.",
    )
    st.stop()

# -----------------------------
# Main page
# -----------------------------
st.title("🛡️ GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Provide URL features in the sidebar and click **Analyze** to begin.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot, caption="Importance of features used by the trained classifier.")
else:
    # -------------------------
    # Step 1: Convert form inputs to model-friendly numeric features
    # -------------------------
    input_dict = {
        'having_IP_Address':  1 if form_values['has_ip'] else -1,
        'URL_Length':         -1 if form_values['url_length'] == 'Short' else (0 if form_values['url_length'] == 'Normal' else 1),
        'Shortining_Service': 1 if form_values['short_service'] else -1,
        'having_At_Symbol':   1 if form_values['at_symbol'] else -1,
        'double_slash_redirecting': -1,  # baseline
        'Prefix_Suffix':      1 if form_values['prefix_suffix'] else -1,
        'having_Sub_Domain':  -1 if form_values['sub_domain'] == 'None' else (0 if form_values['sub_domain'] == 'One' else 1),
        'SSLfinal_State':     -1 if form_values['ssl_state'] == 'None' else (0 if form_values['ssl_state'] == 'Suspicious' else 1),
        'Abnormal_URL':       1 if form_values['abnormal_url'] else -1,
        'URL_of_Anchor': 0, 'Links_in_tags': 0, 'SFH': 0,
        'has_political_keyword': 1 if form_values['has_political_keyword'] else -1,
    }
    input_data = pd.DataFrame([input_dict])

    # -------------------------
    # Risk scoring (visual only)
    # -------------------------
    risk_scores = {
        "Bad SSL":            25 if input_dict['SSLfinal_State'] < 1 else 0,
        "Abnormal URL":       20 if input_dict['Abnormal_URL'] == 1 else 0,
        "Prefix/Suffix":      15 if input_dict['Prefix_Suffix'] == 1 else 0,
        "Shortened URL":      15 if input_dict['Shortining_Service'] == 1 else 0,
        "Complex Sub-domain": 10 if input_dict['having_Sub_Domain'] == 1 else 0,
        "Long URL":           10 if input_dict['URL_Length'] == 1 else 0,
        "Uses IP Address":     5 if input_dict['having_IP_Address'] == 1 else 0,
        "Political Keyword":   5 if input_dict['has_political_keyword'] == 1 else 0,
    }
    risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Feature', 'Risk Contribution']) \
                .sort_values('Risk Contribution', ascending=False)

    # -------------------------
    # Execution workflow
    # -------------------------
    with st.status("Executing SOAR playbook...", expanded=True) as status:
        st.write("▶️ **Step 1: Predictive Analysis** — Running classification model.")
        time.sleep(0.5)
        clf_pred = predict_clf(model, data=input_data)
        is_malicious = clf_pred['prediction_label'].iloc[0] == 1
        verdict = "MALICIOUS" if is_malicious else "BENIGN"
        st.write(f"▶️ **Step 2: Verdict Interpretation** — Model predicts **{verdict}**.")
        time.sleep(0.5)

        # ----- Threat Attribution (only if malicious) -----
        predicted_profile, cluster_id, clu_pred_df = None, None, None
        if is_malicious and cluster_model:
            st.write("▶️ **Step 3: Threat Attribution** — Profiling threat actor via clustering.")
            try:
                clu_pred_df = predict_clu(cluster_model, data=input_data)  # predict_model -> typically 'Label'
                # Prefer 'Label' (predict_model), fall back to 'Cluster' (assign_model), then 'prediction_label'
                label_col = next((c for c in ['Label', 'Cluster', 'prediction_label'] if c in clu_pred_df.columns), None)
                if label_col is None:
                    st.warning(f"Clustering output missing 'Label'/'Cluster'. Columns: {list(clu_pred_df.columns)}")
                else:
                    raw = clu_pred_df[label_col].iloc[0]
                    # Coerce values like 2, "2", "Cluster 2" -> 2
                    val = pd.to_numeric(str(raw).replace('Cluster', '').strip(), errors='coerce')
                    if pd.notna(val):
                        cluster_id = int(val)
                        predicted_profile = CLUSTER_TO_PROFILE.get(cluster_id, "Unknown")
            except Exception as e:
                st.warning(f"Attribution skipped (clustering error): {e}")
        elif is_malicious and not cluster_model:
            st.info("Clustering model not found — train the unsupervised profiler to enable attribution.")

        # ----- Prescriptive plan (only if malicious) -----
        if is_malicious:
            st.write(f"▶️ **Step 4: Prescriptive Analytics** — Engaging {genai_provider} for action plan.")
            try:
                prescription = generate_prescription(genai_provider, dict(input_dict))
                status.update(label="✅ SOAR Playbook Executed Successfully!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Failed to generate prescription: {e}")
                prescription = None
                status.update(label="🚨 Error during GenAI prescription!", state="error")
        else:
            prescription = None
            status.update(label="✅ Analysis Complete. No threat found.", state="complete", expanded=False)

    # -----------------------------
    # Output tabs (Threat Attribution included)
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 **Analysis Summary**",
        "📈 **Visual Insights**",
        "📜 **Prescriptive Plan**",
        "🕵️ **Threat Attribution**"
    ])

    # Tab 1: Summary
    with tab1:
        st.subheader("Verdict and Key Findings")
        if is_malicious:
            st.error("**Prediction: Malicious Phishing URL**", icon="🚨")
        else:
            st.success("**Prediction: Benign URL**", icon="✅")
        st.metric(
            "Malicious Confidence Score",
            f"{clf_pred['prediction_score'].iloc[0]:.2%}" if is_malicious
            else f"{1 - clf_pred['prediction_score'].iloc[0]:.2%}"
        )
        st.caption("This score represents the model's confidence in its prediction.")

    # Tab 2: Visualization
    with tab2:
        st.subheader("Visual Analysis")
        st.write("#### Risk Contribution by Feature")
        st.bar_chart(risk_df.set_index('Feature'))
        st.caption("Shows which input features contributed most to the risk score.")
        if feature_plot:
            st.write("#### Model Feature Importance (Global)")
            st.image(feature_plot, caption="Overall feature importance learned during training.")

    # Tab 3: Prescriptive Plan
    with tab3:
        st.subheader("Actionable Response Plan")
        if prescription:
            st.success("A prescriptive response plan has been generated by the AI.", icon="🤖")
            st.json(prescription, expanded=False)
            st.write("#### Recommended Actions (for Security Analyst)")
            for i, action in enumerate(prescription.get("recommended_actions", []), 1):
                st.markdown(f"**{i}.** {action}")
            st.write("#### Communication Draft (for End-User/Reporter)")
            st.text_area("Draft", prescription.get("communication_draft", ""), height=150)
        else:
            st.info("No prescriptive plan generated because the URL was classified as benign.")

    # Tab 4: Threat Attribution
    with tab4:
        st.subheader("Predicted Threat Actor Profile")
        if not is_malicious:
            st.info("Attribution runs only for **malicious** verdicts.")
        elif cluster_model and (cluster_id is not None):
            st.metric("Cluster ID", f"{cluster_id}")
            st.metric("Actor Profile", predicted_profile or "Unknown")
            st.write("#### Profile Overview")
            st.write(PROFILE_DESCRIPTIONS.get(predicted_profile, "No description available for this profile."))
            st.caption(
                "Note: Cluster→Profile mapping is configurable. Validate on your dataset "
                "(`data/phishing_with_clusters.csv`) and update the mapping at the top of this app."
            )

            # --- Developer Helper: show cluster↔profile purity (if CSV exists)
            if os.path.exists(CLUSTER_CSV):
                with st.expander("🔧 Developer: Show cluster↔actor_profile purity table"):
                    df = pd.read_csv(CLUSTER_CSV)
                    if {'cluster_id', 'actor_profile'}.issubset(df.columns):
                        ct = pd.crosstab(df['cluster_id'], df['actor_profile'], normalize='index').round(2)
                        st.write("Rows = discovered clusters, Columns = ground-truth actor_profile (row-normalized).")
                        st.dataframe(ct)
                        st.caption("Use the dominant column per row to set CLUSTER_TO_PROFILE.")
                    else:
                        st.info("`phishing_with_clusters.csv` present, but missing required columns.")
            else:
                st.caption("Tip: run training to generate `data/phishing_with_clusters.csv` for mapping assistance.")
        elif not cluster_model:
            st.warning("Clustering model not loaded. Train and place `models/threat_actor_profiler.pkl` to enable attribution.")
        else:
            st.warning("Cluster could not be determined for this input.")
            if show_debug and (clu_pred_df is not None):
                st.write("Raw clustering output:")
                st.dataframe(clu_pred_df)
