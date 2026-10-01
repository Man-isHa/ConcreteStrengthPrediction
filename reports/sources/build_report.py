"""Build any HTML report in this folder into a PDF (headless Chrome + page numbers).

There is no LaTeX on this machine, so reports are written as HTML styled to look like
the earlier LaTeX documents and printed with Chrome.

Run from the repository root:
    python reports/sources/build_report.py 03_rcf_final_model_study_le60.html
    python reports/sources/build_report.py 04_rha_vs_rcf_comparison.html

The HTML sources live in reports/sources/; the built PDF is written to reports/.
"""
import os
import subprocess
import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def build(html_name):
    html = HERE / html_name
    pdf = HERE.parent / (html.stem + ".pdf")      # reports/<name>.pdf
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, capture_output=True, timeout=300)
    doc = fitz.open(pdf)
    for i, page in enumerate(doc, 1):
        box = fitz.Rect(0, page.rect.height - 50, page.rect.width, page.rect.height - 22)
        page.insert_textbox(box, str(i), fontname="times-roman", fontsize=11,
                            align=fitz.TEXT_ALIGN_CENTER)
    tmp = pdf.with_suffix(".tmp.pdf")
    doc.save(tmp)
    doc.close()
    os.replace(tmp, pdf)
    print(f"{pdf.name}: {fitz.open(pdf).page_count} pages")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "03_rcf_final_model_study_le60.html")
