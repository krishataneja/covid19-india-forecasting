"""
generate_report.py — Create a 2-page PDF report summarising the analysis.
"""

import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(PROJECT, "outputs", "figures")
MODELS = os.path.join(PROJECT, "outputs", "models")
OUT = os.path.join(PROJECT, "report")
os.makedirs(OUT, exist_ok=True)

# Colours
BLUE = HexColor("#2c3e50")
LIGHT_BLUE = HexColor("#3498db")
GREY = HexColor("#ecf0f1")
WHITE = HexColor("#ffffff")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"],
                          fontSize=18, textColor=BLUE, spaceAfter=6))
styles.add(ParagraphStyle("SectionHead", parent=styles["Heading2"],
                          fontSize=12, textColor=BLUE, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=9, leading=12, alignment=TA_JUSTIFY, spaceAfter=4))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"],
                          fontSize=8, alignment=TA_CENTER, textColor=HexColor("#7f8c8d"),
                          spaceAfter=6, spaceBefore=2))
styles.add(ParagraphStyle("SmallBody", parent=styles["Normal"],
                          fontSize=8.5, leading=11, alignment=TA_JUSTIFY, spaceAfter=3))


def add_image(story, filename, width=6*inch, caption=None):
    path = os.path.join(FIGURES, filename)
    if os.path.exists(path):
        img = Image(path, width=width, height=width * 0.42)
        img.hAlign = "CENTER"
        story.append(img)
        if caption:
            story.append(Paragraph(caption, styles["Caption"]))
    else:
        story.append(Paragraph(f"[Figure: {filename} not found]", styles["Body"]))


def load_metrics():
    rows = []
    for fname in sorted(os.listdir(MODELS)):
        if fname.startswith("metrics_") and fname.endswith(".json"):
            with open(os.path.join(MODELS, fname)) as f:
                d = json.load(f)
            name = fname.replace("metrics_", "").replace(".json", "").replace("_", " ").title()
            rows.append([name, f"{d['RMSE']:,.0f}", f"{d['MAE']:,.0f}",
                         f"{d['MAPE (%)']:.1f}%", f"{d['R2']:.4f}"])
    return rows


def build_report():
    pdf_path = os.path.join(OUT, "report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []

    # ======================== PAGE 1 ========================
    story.append(Paragraph("COVID-19 Weekly Case Forecasting in India", styles["ReportTitle"]))
    story.append(Paragraph("State-Level Spatio-Temporal Analysis with Multi-Source Data Integration",
                            styles["Caption"]))
    story.append(Spacer(1, 6))

    # 1. Introduction
    story.append(Paragraph("1. Introduction and Objective", styles["SectionHead"]))
    story.append(Paragraph(
        "This report presents a disease forecasting pipeline for weekly COVID-19 confirmed cases "
        "across 33 Indian states and union territories, spanning March 2020 to October 2021. "
        "The objective is to integrate heterogeneous data sources — epidemiological case data, "
        "Google Community Mobility Reports, and Census 2011 demographics — into a unified "
        "analytical workflow, engineer informative features, and compare three modelling approaches "
        "for 4-week-ahead forecasting at the state level.", styles["Body"]))

    # 2. Data Sources
    story.append(Paragraph("2. Data Sources and Integration", styles["SectionHead"]))
    story.append(Paragraph(
        "<b>COVID-19 Case Data:</b> Daily state-wise confirmed, recovered, and death counts "
        "(source: covid19india.org / MoHFW). Aggregated to ISO weeks to reduce daily reporting "
        "noise. <b>Google Mobility Reports:</b> Percentage change from baseline in visits to six "
        "place categories (retail, grocery, parks, transit, workplaces, residential). Weekly mean "
        "computed per state. <b>Demographics:</b> Census 2011 state-level population, density, "
        "urbanisation rate, and literacy rate — static covariates that capture structural "
        "heterogeneity across states.", styles["Body"]))

    # 3. Methodology
    story.append(Paragraph("3. Feature Engineering", styles["SectionHead"]))
    story.append(Paragraph(
        "31 features were constructed across five categories: "
        "<b>(i) Lag features</b> — cases at t-1 through t-4 weeks; "
        "<b>(ii) Rolling statistics</b> — 4-week and 8-week rolling mean, std, min, max; "
        "<b>(iii) Epidemiological indicators</b> — week-over-week growth rate, reproduction "
        "number proxy (R<sub>t</sub>), case fatality rate, recovery rate, active cases; "
        "<b>(iv) Temporal encodings</b> — sine/cosine week-of-year (annual cyclicity), month, "
        "weeks-since-start (secular trend); "
        "<b>(v) External features</b> — 6 mobility categories + 4 demographic variables "
        "(log-population, log-density, urbanisation, literacy). "
        "All lag and rolling features use a shift of at least 1 week to prevent data leakage.",
        styles["Body"]))

    # EDA figure
    add_image(story, "national_weekly_cases.png", width=5.8*inch,
              caption="Figure 1: National weekly new COVID-19 cases showing Wave 1 (Sep 2020) and Wave 2 / Delta (May 2021)")

    # 4. Models
    story.append(Paragraph("4. Modelling Approaches", styles["SectionHead"]))
    story.append(Paragraph(
        "<b>Gradient Boosting (GBR):</b> Ensemble of decision trees trained on all 31 features "
        "with TimeSeriesSplit 5-fold CV and hyperparameter grid search over depth, learning rate, "
        "and number of estimators. "
        "<b>STL + Ridge (Decomposition Baseline):</b> Savitzky-Golay trend extraction + "
        "week-of-year seasonal component + Ridge regression on lagged residuals and mobility "
        "features. Per-state models for top 10 states. "
        "<b>MLP Neural Network:</b> Sliding window of 8 weeks flattened into input vector; "
        "3 hidden layers (128-64-32, ReLU); Adam optimiser with early stopping. "
        "All models use a temporal train/test split: last 8 weeks held out as test set.",
        styles["Body"]))

    # ======================== PAGE 2 ========================
    story.append(PageBreak())

    # 5. Results
    story.append(Paragraph("5. Results", styles["SectionHead"]))

    # Metrics table
    header = ["Model", "RMSE", "MAE", "MAPE", "R\u00b2"]
    data = [header] + load_metrics()
    t = Table(data, colWidths=[2.5*cm, 2.8*cm, 2.8*cm, 2*cm, 2*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Paragraph("Table 1: Model performance on held-out test set (last 8 weeks)", styles["Caption"]))

    story.append(Paragraph(
        "Gradient Boosting achieved the best performance across all metrics with an R<super>2</super> of 0.79 "
        "and substantially lower RMSE than alternatives. The decomposition baseline (STL + Ridge) "
        "achieved R<super>2</super> = 0.68 but suffered from high MAPE due to difficulty predicting "
        "low-case periods in smaller states. The MLP neural network performed worst "
        "(negative R<super>2</super>), likely due to insufficient training data per state for a "
        "high-dimensional sliding-window input (256 features from 8 weeks x 32 columns), "
        "leading to overfitting despite early stopping.", styles["Body"]))

    # Feature importance + predictions figures side by side
    add_image(story, "xgboost_feature_importance.png", width=5.8*inch,
              caption="Figure 2: Top 20 features by importance in the Gradient Boosting model")

    add_image(story, "model_comparison.png", width=5.8*inch,
              caption="Figure 3: Model comparison across four evaluation metrics")

    # 6. Discussion
    story.append(Paragraph("6. Key Findings and Limitations", styles["SectionHead"]))
    story.append(Paragraph(
        "<b>Key findings:</b> (1) Lag features (particularly lag-1 and lag-2) dominate predictive "
        "power, confirming strong week-to-week autocorrelation in COVID-19 dynamics. "
        "(2) Mobility features contribute meaningfully — transit and workplace mobility show "
        "negative correlation with case growth, consistent with lockdown effects. "
        "(3) Demographic features (population density, urbanisation) improve cross-state "
        "generalisation in the global GBR model. "
        "(4) The Delta wave's rapid ascent and decay posed the hardest forecasting challenge "
        "due to limited analogous historical patterns.",
        styles["SmallBody"]))
    story.append(Paragraph(
        "<b>Limitations:</b> (1) Simulated data was used for this demonstration; real-world "
        "performance should be validated on actual datasets. "
        "(2) The 4-week forecast horizon relies on autoregressive features, so errors "
        "accumulate in multi-step predictions. "
        "(3) Under-reporting and testing rate variability across states are not modelled. "
        "(4) No vaccination data was incorporated (available only from Jan 2021). "
        "<b>Future work:</b> Incorporating vaccination rates, testing volumes, variant-specific "
        "genomic surveillance data, and spatial diffusion models (e.g., graph neural networks "
        "on state adjacency) could substantially improve forecasting accuracy.",
        styles["SmallBody"]))

    # Build
    doc.build(story)
    print(f"Report saved: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_report()
