from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def _fmt_num(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def _add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 36, 20, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(
    summary_path: str | Path,
    output_path: str | Path,
    charts_dir: str | Path,
    prediction: dict[str, Any] | None = None,
) -> Path:
    summary_path = Path(summary_path)
    output_path = Path(output_path)
    charts_dir = Path(charts_dir)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    comparison = summary.get("model_comparison", {})
    top_features = summary.get("top_features", [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=30,
        title="SmartRoad Automated Traffic Accident Analytics Report",
        author="SmartRoad",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=22, leading=26, spaceAfter=16
    ))
    styles.add(ParagraphStyle(
        name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER,
        fontSize=11, leading=15, textColor=colors.HexColor("#555555")
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontSize=15, leading=18,
        spaceBefore=8, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["Normal"], fontSize=8.5, leading=11
    ))

    story = []
    story.append(Paragraph("SmartRoad - Traffic Accident Analytics", styles["TitleCenter"]))
    story.append(Paragraph("Automated Analytical Report", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} from the configured collision dataset.",
        styles["SubCenter"],
    ))
    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "This report is generated directly by the SmartRoad dashboard as an additional automation feature. "
        "The predictive target is high-severity collision (Fatal or Serious) versus Slight.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Dataset Overview", styles["Section"]))
    kpi_data = [
        ["Metric", "Value"],
        ["Raw rows", _fmt_num(summary.get("rows_raw"))],
        ["Clean rows", _fmt_num(summary.get("rows_clean"))],
        ["Selected model", str(summary.get("selected_model", "-"))],
        ["Positive/high-severity rate", _fmt_pct(metrics.get("positive_rate"))],
        ["Training records", _fmt_num(metrics.get("n_train"))],
        ["Test records", _fmt_num(metrics.get("n_test"))],
    ]
    t = Table(kpi_data, colWidths=[3.5 * inch, 2.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph("2. Model Performance", styles["Section"]))
    model_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "CV F1"]]
    for name, vals in comparison.items():
        model_rows.append([
            name,
            _fmt_pct(vals.get("accuracy")),
            _fmt_pct(vals.get("precision")),
            _fmt_pct(vals.get("recall")),
            _fmt_pct(vals.get("f1")),
            _fmt_pct(vals.get("cv_f1_mean")),
        ])
    mt = Table(model_rows, colWidths=[1.65*inch, .82*inch, .82*inch, .82*inch, .72*inch, .72*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(mt)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Selected model: <b>{summary.get('selected_model', '-')}</b>, based on the highest mean 5-fold cross-validated F1-score.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 14))

    cm_path = charts_dir / "07_confusion_matrix.png"
    if cm_path.exists():
        story.append(Paragraph("3. Confusion Matrix", styles["Section"]))
        story.append(Image(str(cm_path), width=4.8*inch, height=3.9*inch))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"Held-out test metrics for the selected model: accuracy {_fmt_pct(metrics.get('accuracy'))}, "
            f"precision {_fmt_pct(metrics.get('precision'))}, recall {_fmt_pct(metrics.get('recall'))}, "
            f"and F1 {_fmt_pct(metrics.get('f1'))}.",
            styles["Small"],
        ))
        story.append(PageBreak())

    story.append(Paragraph("4. Key Features", styles["Section"]))
    feature_rows = [["Feature", "Mean importance", "Std. deviation"]]
    for item in top_features[:10]:
        feature_rows.append([
            str(item.get("feature", "-")),
            f"{float(item.get('importance_mean', 0)):.5f}",
            f"{float(item.get('importance_std', 0)):.5f}",
        ])
    ft = Table(feature_rows, colWidths=[3.1*inch, 1.55*inch, 1.55*inch])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ft)
    story.append(Spacer(1, 14))

    if prediction:
        story.append(Paragraph("5. Interactive Prediction Result", styles["Section"]))
        pred_rows = [["Prediction field", "Value"]]
        for key, value in prediction.items():
            label = str(key).replace("_", " ").title()
            if key == "probability":
                value = _fmt_pct(value)
            pred_rows.append([label, str(value)])
        pt = Table(pred_rows, colWidths=[3.3*inch, 2.9*inch])
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(pt)
        story.append(Spacer(1, 14))

    story.append(Paragraph("6. Visual Analytics", styles["Section"]))
    image_specs = [
        ("01_accidents_by_hour.png", 6.3*inch, 3.5*inch),
        ("02_severity_distribution.png", 5.8*inch, 3.6*inch),
        ("04_weekday_hour_heatmap.png", 6.8*inch, 3.1*inch),
        ("05_speed_by_severity_boxplot.png", 5.9*inch, 3.7*inch),
        ("06_accident_scatter_map.png", 5.8*inch, 4.2*inch),
        ("08_model_comparison.png", 6.2*inch, 3.4*inch),
    ]
    for idx, (filename, width, height) in enumerate(image_specs):
        p = charts_dir / filename
        if p.exists():
            story.append(Image(str(p), width=width, height=height))
            story.append(Spacer(1, 8))
            if idx in {1, 3, 5}:
                story.append(PageBreak())

    story.append(Paragraph("7. Conclusion", styles["Section"]))
    story.append(Paragraph(
        "SmartRoad combines preprocessing, exploratory analytics, predictive classification, model comparison, "
        "cross-validation, confusion-matrix evaluation, geographic hotspot clustering and interactive prediction. "
        "This automatically generated report packages the current analytical results for review and documentation.",
        styles["BodyText"],
    ))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return output_path
