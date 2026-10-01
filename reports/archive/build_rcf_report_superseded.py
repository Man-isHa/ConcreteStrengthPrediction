"""Build rcf_le60_report.pdf from rcf_le60_report.html (headless Chrome + page numbers).

There is no LaTeX on this machine, so the report is written as HTML styled to look like
the LaTeX reports and printed with Chrome; page numbers are stamped afterwards.

Run: python build_rcf_report.py
"""
import os
import subprocess
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
HTML, PDF = HERE / "rcf_le60_report.html", HERE / "rcf_le60_report.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={PDF}", HTML.as_uri()], check=True,
               capture_output=True, timeout=300)

doc = fitz.open(PDF)
for i, page in enumerate(doc, 1):
    box = fitz.Rect(0, page.rect.height - 50, page.rect.width, page.rect.height - 22)
    page.insert_textbox(box, str(i), fontname="times-roman", fontsize=11,
                        align=fitz.TEXT_ALIGN_CENTER)
tmp = PDF.with_suffix(".tmp.pdf")
doc.save(tmp); doc.close()
os.replace(tmp, PDF)
print(f"{PDF.name}: {fitz.open(PDF).page_count} pages")
