from pathlib import Path
from src.report_generator import generate_pdf_report

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "SmartRoad_Automated_Report.pdf"
SUMMARY = ROOT / "outputs" / "run_summary.json"
CHARTS = ROOT / "outputs"

if __name__ == "__main__":
    path = generate_pdf_report(SUMMARY, OUTPUT, CHARTS)
    print(f"Automated PDF report created: {path}")
