"""
Sample PDF Generator for Indian Supreme Court Judgments
Generates multi-page sample PDFs to validate extraction, OCR fallback, and chunking contracts.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak


def generate_sample_judgment_pdf(output_path: str):
    """Generates a realistic 3-page Indian Supreme Court judgment PDF."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'LegalTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        alignment=1  # Center
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'LegalBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # PAGE 1: Header, Case Number, Parties, Bench
    story.append(Paragraph("IN THE SUPREME COURT OF INDIA", title_style))
    story.append(Paragraph("CIVIL APPELLATE JURISDICTION", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("CIVIL APPEAL NO. 1042 OF 2023", title_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("State of Maharashtra & Ors. ... Appellant(s)", body_style))
    story.append(Paragraph("VERSUS", ParagraphStyle('Versus', parent=title_style, fontSize=10)))
    story.append(Paragraph("ABC Infrastructure Ltd. & Anr. ... Respondent(s)", body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("DATED: 24th February, 2023", body_style))
    story.append(Paragraph("CORAM: Hon'ble Mr. Justice D.Y. Chandrachud, CJI, Hon'ble Mr. Justice P.S. Narasimha", body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("JUDGMENT", heading_style))
    story.append(Paragraph(
        "Dr. Dhananjaya Y. Chandrachud, CJI.\n1. Leave granted. This appeal arises from a final judgment and order dated 15 November 2022 passed by the High Court of Judicature at Bombay in Writ Petition No. 4501 of 2021.",
        body_style
    ))
    story.append(Paragraph(
        "2. The primary question of law involved in this appeal relates to the interpretation of statutory arbitration clauses under the state procurement guidelines.",
        body_style
    ))

    # Page Break to Page 2
    story.append(PageBreak())

    # PAGE 2: Factual Matrix & Submissions
    story.append(Paragraph("FACTS AND BACKGROUND", heading_style))
    story.append(Paragraph(
        "3. The respondent contractor entered into a concession agreement with the appellant on 12 June 2018 for the construction of a state highway bypass. The completion timeline specified a period of thirty-six months from the appointment date.",
        body_style
    ))
    story.append(Paragraph(
        "4. Disputes arose regarding delays in right-of-way handover and environmental clearances. The respondent invoked arbitration under clause 24.3 of the concession agreement.",
        body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("SUBMISSIONS OF COUNSEL", heading_style))
    story.append(Paragraph(
        "5. Learned Senior Counsel appearing on behalf of the appellants contended that the High Court exceeded its jurisdiction under Section 11 of the Arbitration and Conciliation Act, 1996 by delving into arbitrability of non-notified claims.",
        body_style
    ))

    # Page Break to Page 3
    story.append(PageBreak())

    # PAGE 3: Analysis, Operative Order & Signature
    story.append(Paragraph("ANALYSIS AND CONCLUSION", heading_style))
    story.append(Paragraph(
        "6. We have considered the rival submissions and examined the record. The law regarding the scope of pre-referral judicial scrutiny under Section 11(6) is well-settled through the decisions of this Court.",
        body_style
    ))
    story.append(Paragraph(
        "7. For the reasons recorded above, the appeal is allowed. The impugned judgment of the High Court is set aside. Pending application(s), if any, stand disposed of.",
        body_style
    ))
    story.append(Spacer(1, 20))
    story.append(Paragraph("...................................................CJI.\n(Dr. Dhananjaya Y. Chandrachud)", body_style))
    story.append(Paragraph("...................................................J.\n(P.S. Narasimha)", body_style))
    story.append(Paragraph("New Delhi;\nFebruary 24, 2023.", body_style))

    doc.build(story)
    print(f"Generated sample judgment PDF at: {output_path}")


if __name__ == "__main__":
    generate_sample_judgment_pdf("data/sample_pdfs/INSC_2023_CA_1042.pdf")
