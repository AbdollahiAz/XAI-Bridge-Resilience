# XAI_Bridge_Main.py
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from shap import Explanation


def display_figure_as_svg(fig):
    """Render a Matplotlib figure in Streamlit as an SVG vector image."""
    svg_buffer = StringIO()

    fig.savefig(
        svg_buffer,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )

    svg_text = svg_buffer.getvalue()

    st.image(
        svg_text,
        use_container_width=True,
    )


def main():
    # ─── Page configuration ───────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Shapley value estimation for flood-resilient aging bridges",
        layout="wide",
    )

    # Use the folder containing this Python file as the application folder.
    app_dir = Path(__file__).resolve().parent

    # ─── Sidebar logo and input parameters ────────────────────────────────────────
    logo_path = app_dir / "bridge.png"
    if logo_path.exists():
        st.sidebar.image(
            str(logo_path),
            caption="XAI-Bridge-Resilience App",
            width=270,
        )
    else:
        st.sidebar.warning("The file bridge.png was not found.")

    env_labels = {
        1: "Benign",
        2: "Low",
        3: "Moderate",
        4: "Severe",
    }

    with st.sidebar:
        st.header("Input Parameters")

        # 1. Bridge age
        sel_A = st.selectbox(
            r"$\mathrm{Age}\,[\mathrm{year}]$",
            [0, 20, 40, 60, 80, 100],
        )

        # 2. Scour depth: fixed discrete values
        ds_options = [
            0,
            0.25,
            0.50,
            0.75,
            1.00,
            1.25,
            1.50,
            1.75,
            2.00,
            2.50,
            3.00,
            4.00,
        ]

        sel_sc = st.selectbox(
            r"$D_{\mathrm{sc}}\,[\mathrm{m}]$",
            options=ds_options,
            index=0,
            format_func=lambda x: f"{x:g}",
        )

        # 3. Number of spans
        sel_span = st.selectbox(
            r"$\mathrm{Span}\ \#$",
            [1, 3, 6, 9, 12, 15, 18],
        )

        # 4. Post-flood time: dynamically determined by Span #
        max_by_span = {
            1: 150,
            3: 220,
            6: 265,
            9: 290,
            12: 305,
            15: 320,
            18: 335,
        }

        days = [1] + list(range(5, max_by_span[sel_span] + 1, 5))

        sel_day = st.selectbox(
            r"$\mathrm{Time}\,[\mathrm{day}]$",
            options=days,
            index=0,
        )

        # 5. Environmental condition
        env_labels = {
            1: "Benign",
            2: "Low",
            3: "Moderate",
            4: "Severe",
        }

        sel_env = st.selectbox(
            r"$\mathrm{Env}_{\mathrm{cond}}$",
            options=list(env_labels.keys()),
            format_func=lambda x: env_labels[x],
        )

        run = st.button("Run")

    # ─── Main-page header ─────────────────────────────────────────────────────────
    st.title("🔍 Shapley Value Estimation for Flood-Resilient Aging Bridges")

    st.write(
        "✅ Select input parameters to retrieve the true and predicted resilience "
        "indices and the corresponding SHAP waterfall plots. In addition, SHAP "
        "beeswarm plots provide a bird's-eye view of feature importance for the "
        "corresponding age:\n\n"
        "- VAEAC (dependency-aware conditional Shapley value) approach  \n"
        "- Independence (marginal Shapley value) approach"
    )

    # ─── Parameter definitions ────────────────────────────────────────────────────
    st.write("**🔷 Parameter definitions:**")

    cols = st.columns(3)
    labels_row1 = [
        r"$\mathrm{Time}$: Post-flood time [days]",
        r"$D_{\mathrm{sc}}$: Scour depth [m]",
        r"$\mathrm{Span}\,\#$: Number of spans",
    ]

    for col, label in zip(cols, labels_row1):
        col.markdown(f"- {label}")

    st.markdown(
        r"- $\mathrm{Env}_{\mathrm{cond}}$: Environmental condition "
        r"(benign, low, moderate, severe)"
    )

    st.markdown(
        r"- $\Pr(\mathrm{DS}_{\mathrm{no}})$, $\Pr(\mathrm{DS}_{\mathrm{min}})$, "
        r"$\Pr(\mathrm{DS}_{\mathrm{mod}})$, $\Pr(\mathrm{DS}_{\mathrm{ext}})$, and "
        r"$\Pr(\mathrm{DS}_{\mathrm{sev}})$: Probabilities of being in the no-damage, "
        r"minor, moderate, extensive, and severe damage states"
    )

    st.markdown(
        r"- $\mathrm{Rest}_{\mathrm{min}}$, $\mathrm{Rest}_{\mathrm{mod}}$, "
        r"$\mathrm{Rest}_{\mathrm{ext}}$, and $\mathrm{Rest}_{\mathrm{sev}}$: "
        r"Post-flood capacity ratios (PFCRs) associated with the minor, "
        r"moderate, extensive, and severe damage states"
    )

    if not run:
        st.info(
            "⚡ Adjust the parameters on the left, then click **Run** "
            "to see the results."
        )
        st.stop()

    # ─── Load the selected Excel files ────────────────────────────────────────────
    year_file = app_dir / f"Year{sel_A}.xlsx"
    shap_file_va = app_dir / f"SHAP_Final_Year_{sel_A}_Vaeac.xlsx"
    shap_file_ind = app_dir / f"SHAP_Final_Year_{sel_A}_Independence.xlsx"

    required_files = [year_file, shap_file_va, shap_file_ind]
    missing_files = [path.name for path in required_files if not path.exists()]

    if missing_files:
        st.error(
            "The following required files were not found in the application "
            f"folder: {', '.join(missing_files)}"
        )
        return

    try:
        year_df = pd.read_excel(year_file, engine="openpyxl")
        shap_df_va = pd.read_excel(shap_file_va, engine="openpyxl")
        shap_df_ind = pd.read_excel(shap_file_ind, engine="openpyxl")
    except Exception as exc:
        st.error(f"Error loading the Excel files: {exc}")
        return

    # ─── Validate required data columns ───────────────────────────────────────────
    required_year_columns = {
        "$Env_{cond}$",
        "Span #",
        "Time",
        "$D_{sc}$",
    }
    missing_year_columns = required_year_columns.difference(year_df.columns)

    if missing_year_columns:
        st.error(
            "The Year file is missing the following columns: "
            + ", ".join(sorted(missing_year_columns))
        )
        return

    if "base" not in shap_df_va.columns or "base" not in shap_df_ind.columns:
        st.error("Both SHAP files must contain a column named 'base'.")
        return

    # Keep the original Excel headers for reading values, but use clean
    # mathematical labels only for displaying the SHAP plots.
    shap_feature_columns = [
        column for column in shap_df_va.columns if column != "base"
    ]

    if not shap_feature_columns:
        st.error("No feature columns were found in the VAEAC SHAP file.")
        return

    missing_in_ind = [
        feature
        for feature in shap_feature_columns
        if feature not in shap_df_ind.columns
    ]

    if missing_in_ind:
        st.error(
            "The Independence SHAP file is missing these feature columns: "
            + ", ".join(missing_in_ind)
        )
        return

    # The SHAP spreadsheets may contain older LaTeX column names that do not
    # exactly match the headers in the Year spreadsheet. The model features are
    # therefore aligned by their established column order.
    target_col = year_df.columns[-1]
    year_feature_columns = [
        column for column in year_df.columns if column != target_col
    ]

    if len(year_feature_columns) != len(shap_feature_columns):
        st.error(
            "The Year file and SHAP files do not contain the same number of "
            f"features. Year file: {len(year_feature_columns)}; "
            f"SHAP file: {len(shap_feature_columns)}."
        )
        st.write("Year feature columns:", year_feature_columns)
        st.write("SHAP feature columns:", shap_feature_columns)
        return

    # Clean labels shown in waterfall and beeswarm plots.
    display_feature_names = [
        r"$\mathrm{Time}$",
        r"$D_{\mathrm{sc}}$",
        r"$\mathrm{Span}\,\#$",
        r"$\mathrm{Env}_{\mathrm{cond}}$",
        r"$\Pr(\mathrm{DS}_{\mathrm{no}})$",
        r"$\Pr(\mathrm{DS}_{\mathrm{min}})$",
        r"$\Pr(\mathrm{DS}_{\mathrm{mod}})$",
        r"$\Pr(\mathrm{DS}_{\mathrm{ext}})$",
        r"$\Pr(\mathrm{DS}_{\mathrm{sev}})$",
        r"$\mathrm{Rest}_{\mathrm{min}}$",
        r"$\mathrm{Rest}_{\mathrm{mod}}$",
        r"$\mathrm{Rest}_{\mathrm{ext}}$",
        r"$\mathrm{Rest}_{\mathrm{sev}}$",
    ]

    if len(display_feature_names) != len(shap_feature_columns):
        st.error(
            "The number of display labels does not match the number of "
            "model features."
        )
        return

    # ─── Find the row matching the selected parameters ────────────────────────────
    mask = (
        (year_df["$Env_{cond}$"] == sel_env)
        & (year_df["Span #"] == sel_span)
        & (year_df["Time"] == sel_day)
        & np.isclose(
            year_df["$D_{sc}$"].astype(float),
            float(sel_sc),
            rtol=0.0,
            atol=1.0e-10,
        )
    )

    matches = year_df.loc[mask]

    if matches.empty:
        st.warning("No matching row was found in the selected Year file.")
        return

    if len(matches) > 1:
        st.warning(
            f"{len(matches)} matching rows were found. "
            "The first matching row will be used."
        )

    row_idx = int(matches.index[0])

    # ─── Validate row alignment across the three Excel files ──────────────────────
    if not (0 <= row_idx < len(shap_df_va)):
        st.error("The matching row index is outside the VAEAC SHAP file.")
        return

    if not (0 <= row_idx < len(shap_df_ind)):
        st.error("The matching row index is outside the Independence SHAP file.")
        return

    # ─── Retrieve the ground-truth resilience value ───────────────────────────────
    true_val = pd.to_numeric(
        pd.Series([matches.iloc[0][target_col]]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(true_val):
        st.error(
            f"The target value in column '{target_col}' is not numeric."
        )
        return

    # ─── Prepare feature values for the selected row ──────────────────────────────
    selected_features = matches.loc[:, year_feature_columns].iloc[0]
    feat_values = selected_features.to_numpy()[np.newaxis, :]

    # ─── Build a one-row SHAP Explanation ─────────────────────────────────────────
    def make_row_explanation(shap_df):
        base_value = float(shap_df.at[row_idx, "base"])
        shap_values = (
            shap_df.loc[row_idx, shap_feature_columns]
            .astype(float)
            .to_numpy()
        )

        return Explanation(
            values=shap_values[np.newaxis, :],
            base_values=np.array([base_value]),
            data=feat_values,
            feature_names=display_feature_names,
        )

    try:
        expl_va_row = make_row_explanation(shap_df_va)
        expl_ind_row = make_row_explanation(shap_df_ind)
    except Exception as exc:
        st.error(f"Error creating the row-level SHAP explanations: {exc}")
        return

    # ─── Show the ground-truth and predicted resilience values ────────────────────
    predicted_val = (
        float(np.asarray(expl_va_row.base_values).reshape(-1)[0])
        + float(np.asarray(expl_va_row.values).sum())
    )

    st.subheader("Resilience index")

    st.markdown(f"- **Ground truth value = {true_val:.6f}**")
    st.markdown(f"- **Predicted value = {predicted_val:.6f}**")

    # ─── Plotting settings ────────────────────────────────────────────────────────
    tick_font_size = 12
    annotation_font_size = 12
    x_label_font_size = 13

    def render_waterfall(explanation, title):
        st.subheader(title)

        plt.figure(figsize=(8, 7))
        shap.plots.waterfall(
            explanation[0],
            max_display=len(display_feature_names),
            show=False,
        )

        fig = plt.gcf()
        ax = plt.gca()

        ax.tick_params(axis="y", labelsize=tick_font_size)
        ax.tick_params(axis="x", labelsize=tick_font_size)
        ax.set_xlabel(
            "Shapley value (impact on the resilience index)",
            fontsize=x_label_font_size,
            labelpad=15,
        )

        for text_item in ax.texts:
            text_item.set_fontsize(annotation_font_size)

        fig.tight_layout()
        display_figure_as_svg(fig)
        plt.close(fig)

    # ─── Render waterfall plots ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        render_waterfall(expl_va_row, "VAEAC – Waterfall")

    with col2:
        render_waterfall(expl_ind_row, "Independence – Waterfall")

    # ─── Build full-data SHAP Explanations for beeswarm plots ─────────────────────
    bees_data = year_df.loc[:, year_feature_columns].copy()

    # Keep all continuous/numerical features numeric.
    for column in bees_data.columns:
        bees_data[column] = pd.to_numeric(bees_data[column], errors="coerce")

    if bees_data.isna().any().any():
        st.error(
            "At least one feature column in the Year file contains a "
            "non-numeric or missing value."
        )
        return

    # Display the environmental-condition feature as a categorical variable.
    # SHAP renders categorical/string feature values in gray in beeswarm plots.
    # Assign by column name rather than .iloc so pandas can safely change the
    # column dtype from numeric to object/string.
    env_feature_position = 3
    env_feature_column = year_feature_columns[env_feature_position]

    env_display_values = (
        pd.to_numeric(
            bees_data[env_feature_column],
            errors="coerce",
        )
        .astype("Int64")
        .map(env_labels)
    )

    if env_display_values.isna().any():
        st.error(
            "The environmental-condition column contains values outside "
            "the expected codes 1, 2, 3, and 4."
        )
        return

    bees_data[env_feature_column] = env_display_values.astype(object)

    try:
        va_values = (
            shap_df_va.loc[:, shap_feature_columns]
            .astype(float)
            .to_numpy()
        )
        ind_values = (
            shap_df_ind.loc[:, shap_feature_columns]
            .astype(float)
            .to_numpy()
        )
        va_base_values = shap_df_va["base"].astype(float).to_numpy()
        ind_base_values = shap_df_ind["base"].astype(float).to_numpy()
    except Exception as exc:
        st.error(f"SHAP values or base values are not numeric: {exc}")
        return

    if len(year_df) != len(shap_df_va) or len(year_df) != len(shap_df_ind):
        st.error(
            "The Year, VAEAC SHAP, and Independence SHAP files must "
            "contain the same number of rows."
        )
        return

    expl_va_all = Explanation(
        values=va_values,
        base_values=va_base_values,
        data=bees_data.to_numpy(),
        feature_names=display_feature_names,
    )

    expl_ind_all = Explanation(
        values=ind_values,
        base_values=ind_base_values,
        data=bees_data.to_numpy(),
        feature_names=display_feature_names,
    )

    def render_beeswarm(explanation, title):
        st.subheader(title)

        plt.figure(figsize=(8, 7))
        shap.plots.beeswarm(
            explanation,
            max_display=len(display_feature_names),
            show=False,
        )

        fig = plt.gcf()
        ax = plt.gca()

        ax.tick_params(axis="y", labelsize=tick_font_size)
        ax.tick_params(axis="x", labelsize=tick_font_size)
        ax.set_xlabel(
            "Shapley value (impact on the resilience index)",
            fontsize=x_label_font_size,
            labelpad=12,
        )

        fig.tight_layout()
        display_figure_as_svg(fig)
        plt.close(fig)

    # ─── Render beeswarm plots ────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        render_beeswarm(expl_va_all, "VAEAC – Beeswarm")

    with col4:
        render_beeswarm(expl_ind_all, "Independence – Beeswarm")

    # ─── Disclaimer ───────────────────────────────────────────────────────────────
    st.markdown(
        """
📜 **Disclaimer:**

- This work is part of a manuscript titled
  ***Explainable AI reveals dependence modeling can outweigh aging in
  infrastructure resilience assessment***. It has not yet been peer reviewed,
  and any reference to this work should be postponed until after its acceptance.
- For questions, please contact the developers:
  - Ali Amini ([ali.amini@mail.mcgill.ca](mailto:ali.amini@mail.mcgill.ca))
  - Azam Abdollahi
    ([azam.abdollahi2024@gmail.com](mailto:azam.abdollahi2024@gmail.com))
  - Yazhou (Tim) Xie ([tim.xie@mcgill.ca](mailto:tim.xie@mcgill.ca))
  - Sotirios A. Argyroudis
    ([sotirios.argyroudis@brunel.ac.uk](mailto:sotirios.argyroudis@brunel.ac.uk))
  - Stergios A. Mitoulis
    ([s.mitoulis@ucl.ac.uk](mailto:s.mitoulis@ucl.ac.uk))
"""
    )


if __name__ == "__main__":
    main()
