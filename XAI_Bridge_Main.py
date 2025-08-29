# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap import Explanation

def main():
    # ─── Page config ──────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Shapley value estimation for flood-resilient aging bridges",
        layout="wide"
    )

    # ─── Sidebar logo + inputs ────────────────────────────────────────────────────
    st.sidebar.image("bridge.png", caption="XAI-Bridge-Resilience App", width=270)
    with st.sidebar:
        st.header("Input Parameters")
        sel_A = st.selectbox(r"$\text{Age}\,[\text{year}]$", [0, 20, 40, 60, 80, 100])


        env_labels = {1: "Benign", 2: "Low", 3: "Moderate", 4: "Severe"}
        sel_env = st.selectbox(
            "$Env_{cond}$",
            options=list(env_labels),
            format_func=lambda x: env_labels[x]
        )

        sel_span = st.selectbox("$Span$ #", [1, 3, 6, 9, 12, 15, 18])
        max_by_span = {1:150, 3:220, 6:265, 9:290, 12:305, 15:320, 18:335}
        days = [1] + list(range(5, max_by_span[sel_span] + 1, 5))
       # sel_day = st.selectbox("$Time$", days)
        sel_day = st.selectbox(r"$Time\,[\mathrm{day}]$", days)
        # ─── Dynamic D_s list ──────────────────────────────────────────────────────
        ds_base = [0,0.25,0.5, 0.75,1, 1.25,1.5,1.75, 2, 2.5, 3, 4]
        ds_options =  ds_base
        sel_sc = st.selectbox(r"$D_{sc}\,[\mathrm{m}]$", ds_options)


    # ─── Main page header ─────────────────────────────────────────────────────────
    st.title("🔍 Shapley Value Estimation for Flood-Resilient Aging Bridges")
    st.write("**💻 Developers: Ali Amini, Azam Abdollahi, Sotirios A. Argyroudis, Yazhou (Tim) Xie, and Stergios A. Mitoulis**")
    st.write(
        "✅ Select input parameters to retrieve the true and predicted resilience indices and the corresponding SHAP waterfall plots. In addition, SHAP beeswarm plots offer bird's eye view of feature importance for the corresponding age:\n\n"
        "- VAEAC (dependency-aware conditional Shapley value ) approach  \n"
        "- Independence approach (marginal Shapley value)"
    )

  # ─── Four bullet-rows ──────────────────────────────────────────────────────────
    st.write("**🔷 Parameter definition:**")
    cols = st.columns(3)
    labels_row1 = [
        "$Time$: Post-flood time [days] ",
        "$D_{sc}$: Scour depth [m]",
        r"$Span\#$: Number of spans"
    ]
    for col, lbl in zip(cols, labels_row1):
        col.markdown(f"- {lbl}")

    # Row 2: environment condition
    st.markdown(
        "- $Env_{cond}$: Environmental condition "
        "(benign, low, moderate, severe)"
    )

    # Row 3: damage-state probabilities
    st.markdown(
        "- $DS_{no}$, $DS_{min}$, $DS_{mod}$, $DS_{ext}$, $DS_{sev}$: "
        "Probability of being in no, minor, moderate, extensive, "
        "and severe damage states"
    )

    # Row 4: restoration ratios
    st.markdown(
        "- $Rest_{min}$, $Rest_{mod}$, $Rest_{ext}$, $Rest_{sev}$: "
        "Post-flood capacity ratio (PFCR) at minor, moderate, "
        "extensive, and severe damage states"
    )


    run = st.sidebar.button("Run")
    if not run:
        st.info("⚡ Adjust parameters on the left, then click **Run** to see results.")
        st.stop()

    # ─── Load Excel files ─────────────────────────────────────────────────────────
    year_file     = f"Year{sel_A}.xlsx"
    shap_file_va  = f"SHAP_Final_Year_{sel_A}_Vaeac.xlsx"
    shap_file_ind = f"SHAP_Final_Year_{sel_A}_Independence.xlsx"

    try:
        year_df     = pd.read_excel(year_file, engine="openpyxl")
        shap_df_va  = pd.read_excel(shap_file_va, engine="openpyxl")
        shap_df_ind = pd.read_excel(shap_file_ind, engine="openpyxl")
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    # ─── Filter for matching row ──────────────────────────────────────────────────
    mask = (
        (year_df["$Env_{cond}$"] == sel_env) &
        (year_df["Span #"]      == sel_span) &
        (year_df["Time"]         == sel_day)  &
        (year_df["$D_{sc}$"]       == sel_sc)
    )
    matches = year_df[mask]
    if matches.empty:
        st.warning("No matching row found in Year file.")
        return

    # ─── Show true value ──────────────────────────────────────────────────────────
    st.subheader("True resilience index")
    target_col = year_df.columns[-1]
    true_val   = matches.iloc[0][target_col]
    st.write(f"**{target_col} = {true_val:.6f}**")

    # ─── Determine SHAP row index ─────────────────────────────────────────────────
    row_idx = matches.index[0]
    if not (0 <= row_idx < len(shap_df_va) and 0 <= row_idx < len(shap_df_ind)):
        st.error("Row index out of bounds in SHAP files.")
        return

    # ─── Prepare features and values ──────────────────────────────────────────────
    feat_names  = [c for c in shap_df_va.columns if c != "base"]
    feat_values = matches[feat_names].iloc[0].values[np.newaxis, :]

    # ─── Build per-row Explanation for waterfalls ─────────────────────────────────
    def make_row_expl(shap_df):
        base    = shap_df.at[row_idx, "base"]
        vals    = shap_df.drop(columns=["base"]).iloc[row_idx].values
        return Explanation(
            values        = vals[np.newaxis, :],
            base_values   = np.array([base]),
            data          = feat_values,
            feature_names = feat_names
        )

    expl_va_row  = make_row_expl(shap_df_va)
    expl_ind_row = make_row_expl(shap_df_ind)

    # ─── Render WATERFALL plots ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    TICK_FS, ANNOT_FS, XLABEL_FS = 14, 14, 16

    with col1:
        st.subheader("VAEAC – Waterfall")
        total_va = shap_df_va.iloc[row_idx].sum()
        st.write(f"**Predicted resilience = {total_va:.6f}**")
        fig1, ax1 = plt.subplots(figsize=(3,2), dpi=100)
        shap.plots.waterfall(expl_va_row[0], max_display=len(feat_names))
        ax1.tick_params(axis='y', labelsize=TICK_FS)
        ax1.tick_params(axis='x', labelsize=TICK_FS)
        ax1.set_xlabel("Shapley value (impact on the resilience index)", fontsize=XLABEL_FS, labelpad=15)
        for t in ax1.texts: t.set_fontsize(ANNOT_FS)
        st.pyplot(fig1)

    with col2:
        st.subheader("Independence – Waterfall")
        total_ind = shap_df_ind.iloc[row_idx].sum()
        st.write(f"**Predicted resilience = {total_ind:.6f}**")
        fig2, ax2 = plt.subplots(figsize=(3,2), dpi=100)
        shap.plots.waterfall(expl_ind_row[0], max_display=len(feat_names))
        ax2.tick_params(axis='y', labelsize=TICK_FS)
        ax2.tick_params(axis='x', labelsize=TICK_FS)
        ax2.set_xlabel("Shapley value (impact on the resilience index)", fontsize=XLABEL_FS, labelpad=15)
        for t in ax2.texts: t.set_fontsize(ANNOT_FS)
        st.pyplot(fig2)

    # ─── Build full-dataset Explanations for beeswarms ────────────────────────────
    bees_data = year_df[feat_names].copy()
    bees_data["$Env_{cond}$"] = bees_data["$Env_{cond}$"].map(env_labels)

    expl_va_all = Explanation(
        values        = shap_df_va.drop(columns=["base"]).values,
        base_values   = shap_df_va["base"].values,
        data          = bees_data.values,
        feature_names = feat_names
    )
    expl_ind_all = Explanation(
        values        = shap_df_ind.drop(columns=["base"]).values,
        base_values   = shap_df_ind["base"].values,
        data          = bees_data.values,
        feature_names = feat_names
    )

    # ─── Render BEESWARM plots ────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("VAEAC – Beeswarm")
        fig3, ax3 = plt.subplots(figsize=(3,2), dpi=100)
        shap.plots.beeswarm(expl_va_all, max_display=len(feat_names))
        ax3.tick_params(axis='y', labelsize=TICK_FS)
        ax3.tick_params(axis='x', labelsize=TICK_FS)
        ax3.set_xlabel("Shapley value (impact on the resilience index)", fontsize=XLABEL_FS, labelpad=12)
        st.pyplot(fig3)

    with col4:
        st.subheader("Independence – Beeswarm")
        fig4, ax4 = plt.subplots(figsize=(3,2), dpi=100)
        shap.plots.beeswarm(expl_ind_all, max_display=len(feat_names))
        ax4.tick_params(axis='y', labelsize=TICK_FS)
        ax4.tick_params(axis='x', labelsize=TICK_FS)
        ax4.set_xlabel("Shapley value (impact on the resilience index)", fontsize=XLABEL_FS, labelpad=12)
        st.pyplot(fig4)



    st.markdown(
        """
    📜**Disclaimer:**

    - This work is part of a manuscript titled ***Accuracy–Efficiency Trade-off for Data-Driven Explainability: Flood-Resilient Aging Bridges*** submitted to *Nature Sustainability*. It has not yet been peer reviewed, and any reference to this work should be postponed until after its acceptance.
    - If you have any questions, please contact:
        - A. Amini ([ali.amini@mail.mcgill.ca](mailto:ali.amini@mail.mcgill.ca))
        - A. Abdollahi ([azam.abdollahi2024@gmail.com](mailto:azam.abdollahi2024@gmail.com))
        - S. A. Argyroudis ([sotirios.argyroudis@brunel.ac.uk](mailto:sotirios.argyroudis@brunel.ac.uk))
        - Y. Xie ([tim.xie@mcgill.ca](mailto:tim.xie@mcgill.ca))
        - S. A. Mitoulis ([s.mitoulis@ucl.ac.uk](mailto:s.mitoulis@ucl.ac.uk))
        """
    )




if __name__ == "__main__":
    main()
