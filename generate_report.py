"""
generate_report.py
------------------
Generate a multi-page PDF report covering all experiments in the
Hydra Effect Refusal Mechanistic Interpretability project.

Usage:
    cd /workspace/hydra-effect-refusal
    python generate_report.py
    -> Writes /workspace/hydra_effect_report.pdf
"""

import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_PATH = "/workspace/hydra_effect_report.pdf"
CACHE_DIR   = "/workspace/cache"

PAGE_W, PAGE_H = A4
MARGIN         = 2.0 * cm
BODY_W         = PAGE_W - 2 * MARGIN

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL   = colors.HexColor("#21908C")
RED    = colors.HexColor("#d44842")
DARK   = colors.HexColor("#1A1A2E")
LIGHT  = colors.HexColor("#F4F4F8")
GRAY   = colors.HexColor("#888888")
WHITE  = colors.white
GOLD   = colors.HexColor("#E8B84B")


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

def S(name, **kw):
    """Return a ParagraphStyle extending base_styles['Normal']."""
    parent = base_styles.get(name, base_styles["Normal"])
    return ParagraphStyle(name + str(id(kw)), parent=parent, **kw)


TITLE_STYLE = S("Normal",
    fontName="Helvetica-Bold", fontSize=28, leading=34,
    textColor=WHITE, alignment=TA_CENTER)

SUBTITLE_STYLE = S("Normal",
    fontName="Helvetica", fontSize=14, leading=18,
    textColor=colors.HexColor("#CCCCEE"), alignment=TA_CENTER)

AUTHOR_STYLE = S("Normal",
    fontName="Helvetica-Oblique", fontSize=11, leading=14,
    textColor=colors.HexColor("#AAAACC"), alignment=TA_CENTER)

H1 = S("Normal",
    fontName="Helvetica-Bold", fontSize=18, leading=22,
    textColor=DARK, spaceBefore=16, spaceAfter=6)

H2 = S("Normal",
    fontName="Helvetica-Bold", fontSize=14, leading=18,
    textColor=TEAL, spaceBefore=12, spaceAfter=4)

H3 = S("Normal",
    fontName="Helvetica-Bold", fontSize=11, leading=14,
    textColor=DARK, spaceBefore=8, spaceAfter=3)

BODY = S("Normal",
    fontName="Helvetica", fontSize=10, leading=14,
    textColor=DARK, alignment=TA_JUSTIFY)

BODY_SMALL = S("Normal",
    fontName="Helvetica", fontSize=9, leading=12,
    textColor=DARK)

MONO = S("Normal",
    fontName="Courier", fontSize=9, leading=12,
    textColor=DARK)

CAPTION = S("Normal",
    fontName="Helvetica-Oblique", fontSize=9, leading=12,
    textColor=GRAY, alignment=TA_CENTER)

FINDING_STYLE = S("Normal",
    fontName="Helvetica", fontSize=10, leading=14,
    textColor=DARK, leftIndent=12, firstLineIndent=-12)

CALLOUT = S("Normal",
    fontName="Helvetica", fontSize=10, leading=14,
    textColor=WHITE, backColor=TEAL,
    borderPadding=(6, 10, 6, 10), alignment=TA_CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
def draw_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Accent strip
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H * 0.38, PAGE_W, 4, fill=1, stroke=0)
    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(DARK)
    canvas.rect(0, PAGE_H - 1.5*cm, PAGE_W, 1.5*cm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, PAGE_H - 1.0*cm,
                      "Hydra Effect — Refusal Mechanism Analysis")
    # Footer
    canvas.setFillColor(LIGHT)
    canvas.rect(0, 0, PAGE_W, 1.2*cm, fill=1, stroke=0)
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, 0.5*cm, "google/gemma-2-2b-it | GemmaScope SAEs")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5*cm, f"Page {doc.page}")
    # Top accent line
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 1.5*cm - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hr(color=TEAL, width=1):
    return HRFlowable(width="100%", thickness=width, color=color, spaceAfter=4)


def img(path, width=BODY_W, caption=None):
    """Return [Image, Caption] or [] if file missing."""
    if not os.path.exists(path):
        return [Paragraph(f"<i>[Image not found: {os.path.basename(path)}]</i>", CAPTION)]
    from reportlab.lib.utils import ImageReader
    ir = ImageReader(path)
    iw, ih = ir.getSize()
    height = width * ih / iw if iw else width * 0.6
    items = [Image(path, width=width, height=height)]
    if caption:
        items.append(Paragraph(caption, CAPTION))
    return items


def finding_row(number, title, body, color=TEAL):
    """Numbered finding block."""
    badge = Table(
        [[Paragraph(f"<b>{number}</b>", S("Normal",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=WHITE, alignment=TA_CENTER))]],
        colWidths=[0.8*cm], rowHeights=[0.8*cm],
        style=[("BACKGROUND", (0,0), (-1,-1), color),
               ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
               ("ALIGN",      (0,0), (-1,-1), "CENTER"),
               ("GRID",       (0,0), (-1,-1), 0, color),
               ("ROUNDEDCORNERS", [4])],
    )
    text = Paragraph(f"<b>{title}</b><br/>{body}", BODY_SMALL)
    row = Table([[badge, text]], colWidths=[1.1*cm, BODY_W - 1.1*cm],
                style=[("VALIGN", (0,0), (-1,-1), "TOP"),
                       ("LEFTPADDING",  (1,0), (1,0), 6),
                       ("BOTTOMPADDING",(0,0), (-1,-1), 4)])
    return row


def results_table(headers, rows, col_widths=None):
    """Styled results table."""
    data = [headers] + rows
    style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1,  0),  DARK),
        ("TEXTCOLOR",    (0, 0), (-1,  0),  WHITE),
        ("FONTNAME",     (0, 0), (-1,  0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1,  0),  9),
        ("FONTNAME",     (0, 1), (-1, -1),  "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1),  9),
        ("BACKGROUND",   (0, 2), (-1, -1),  colors.HexColor("#F4F4F8")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID",         (0, 0), (-1, -1),  0.4, GRAY),
        ("VALIGN",       (0, 0), (-1, -1),  "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1),  5),
        ("RIGHTPADDING", (0, 0), (-1, -1),  5),
        ("TOPPADDING",   (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING",(0, 0), (-1, -1),  4),
    ])
    if col_widths is None:
        col_widths = [BODY_W / len(headers)] * len(headers)
    return Table(data, colWidths=col_widths, style=style, repeatRows=1)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_cover():
    story = []
    story.append(Spacer(1, PAGE_H * 0.15))
    story.append(Paragraph(
        "The Hydra Effect in LLM Refusal", TITLE_STYLE))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Mechanistic Interpretability of Refusal and Jailbreak Mechanisms<br/>"
        "using Sparse Autoencoders on Gemma-2-2B-IT",
        SUBTITLE_STYLE))
    story.append(Spacer(1, 1.0*cm))
    story.append(Paragraph("google/gemma-2-2b-it  ·  GemmaScope SAEs (width 65k)  ·  March 2026",
                            AUTHOR_STYLE))
    return story


def build_overview():
    story = []
    story.append(Paragraph("Project Overview", H1))
    story.append(hr())
    story.append(Paragraph(
        "This report documents a mechanistic interpretability investigation into the "
        "<b>Hydra Effect</b> — the phenomenon where language models maintain refusal "
        "behaviour even after individual refusal features are ablated. Using GemmaScope "
        "Sparse Autoencoders (65,536 features per layer) on <b>google/gemma-2-2b-it</b>, "
        "we dissect how refusal and jailbreak suppression work at the feature level across "
        "layers 9–15 of the residual stream.", BODY))
    story.append(Spacer(1, 0.3*cm))

    # Pipeline table
    story.append(Paragraph("Experiment Pipeline", H2))
    tbl = results_table(
        ["Experiment", "Purpose", "Key Output"],
        [
            ["01 — Downstream", "L14–16 clamping/ablation of refusal features",
             "Hydra Effect confirmed downstream"],
            ["02 — Upstream", "L9–13 harm sensor experiments",
             "Upstream is the true gatekeeper"],
            ["03 — Cross-Prompt", "Generalisation across prompts/jailbreaks",
             "Single jailbreak found (DAN_Roleplay)"],
            ["04 — Jailbreak Suppression", "Full 65k attribution on DAN+bomb prompt",
             "Redundant gatekeeping discovered"],
            ["05 — Request Sensitivity", "Fixed DAN wrapper, 8 swapped requests",
             "Features wrapper- and content-sensitive"],
            ["06 — Universal Bank", "Compliance direction + harmbench attribution bank",
             "45% jailbreak block, 100% specificity"],
            ["07 — Pattern Analysis", "Per-prompt outcomes: refused vs complied",
             "Category-level coverage patterns"],
        ],
        col_widths=[3.5*cm, 7.5*cm, 5.5*cm],
    )
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # Model / SAE info
    story.append(Paragraph("Model & SAE Details", H2))
    tbl2 = results_table(
        ["Component", "Details"],
        [
            ["Model",        "google/gemma-2-2b-it (instruction-tuned, 2B params)"],
            ["SAEs",         "GemmaScope — gemma-scope-2b-pt-res, width_65k (65,536 features/layer)"],
            ["Framework",    "TransformerLens (HookedTransformer) + sae_lens v6+"],
            ["Hardware",     "Runpod GPU, CUDA, bfloat16 model / float32 SAE weights"],
            ["Layers scanned", "L9–15 (upstream L9–13, downstream L14–15)"],
            ["Attribution",  "feat_act × feat_grad at last token; loss = logit for 'I'"],
        ],
        col_widths=[4*cm, 12.5*cm],
    )
    story.append(tbl2)
    return story


def build_key_findings():
    story = []
    story.append(Paragraph("Key Findings", H1))
    story.append(hr())
    story.append(Paragraph(
        "Nine findings emerged across the six experiments, building a mechanistic picture "
        "of refusal, the Hydra Effect, and jailbreak suppression:", BODY))
    story.append(Spacer(1, 0.3*cm))

    findings = [
        ("Upstream is the True Gatekeeper",
         "Clamping top-40 harm features in L9–13 on a direct harm prompt causes the model "
         "to comply. Upstream harm sensors gate all downstream refusal machinery."),
        ("The Hydra Effect",
         "Ablating or clamping any number of downstream refusal features (L14–16) on a "
         "direct harm prompt — the model still refuses. Backup heads compensate because "
         "they are all driven by the same upstream signal."),
        ("Why Upstream Ablation Defeats the Hydra",
         "Removing the upstream harm signal starves ALL downstream heads simultaneously, "
         "including backup heads. No signal → no Hydra."),
        ("Jailbreaks Use Redundant Gatekeeping",
         "DAN_Roleplay activates compliance drivers in BOTH upstream (L9–13) AND downstream "
         "(L14–15). Ablating either region alone restores refusal — the jailbreak only works "
         "by suppressing both simultaneously."),
        ("Cosine-Similarity Bank ≠ Causal Features",
         "The cosine-similarity bank captures only ~3% (Jaccard) of features actually causal "
         "on a jailbreak prompt (Exp 04 overlap). Full 65k attribution is required."),
        ("Compliance Features Have No Hydra Effect",
         "Refusal features are redundant (RLHF-trained robustness). Compliance features are "
         "fragile and non-redundant — ablating downstream compliance drivers alone immediately "
         "restores refusal. Defense only needs to suppress compliance features."),
        ("Wrapper-Sensitive in Aggregate, Content-Sensitive in Identity",
         "Total compliance energy is similar across harmful and benign requests in the same "
         "DAN wrapper (ratio 1.08×). But different requests activate different feature subsets. "
         "Bomb bank overlaps only 45% with meth, 40% with malware, 16% with stalking."),
        ("Compliance Direction is Geometrically Real",
         "compliance_vec = mean(DAN_acts) − mean(direct_acts) at L15 (norm=164). "
         "L15 feat 46131 is rank #2 by BOTH mean attribution (harmbench) AND cosine-similarity "
         "to compliance direction — strong convergent validation (Exp 05 & 06)."),
        ("Universal Compliance Bank: 45% Coverage, 100% Specificity",
         "Bank of 40 features (L9–15) built from 100 harmbench prompts: harmbench+DAN "
         "10%→50% refusal (+40pp); jailbreakbench+DAN 5%→45% (+40pp); benign 0%→0% "
         "(perfect specificity); benign+DAN 0%→0%. L15 feat 46131 excluded by benign "
         "filter (fires on benign = general helpfulness feature, not jailbreak-specific)."),
    ]

    for i, (title, body) in enumerate(findings, 1):
        story.append(finding_row(i, title, body))
        story.append(Spacer(1, 0.25*cm))

    return story


def build_exp01_02():
    story = []
    story.append(Paragraph("Experiments 01 & 02 — Hydra Effect Discovery", H1))
    story.append(hr())

    story.append(Paragraph("Experiment 01 — Downstream Clamping/Ablation", H2))
    story.append(Paragraph(
        "Systematic clamping and ablation of top refusal features in layers 14–16 on a direct "
        "harmful prompt. Despite removing or suppressing many refusal features, the model "
        "continued to refuse — compensating via backup heads.", BODY))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Experiment 02 — Upstream Harm Sensors", H2))
    story.append(Paragraph(
        "Targeted ablation of top harm features in layers 9–13. Clamping top-40 harm features "
        "caused the model to comply with a direct harm request. This confirmed that upstream "
        "harm sensing is the <i>necessary and sufficient</i> gate for refusal.", BODY))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Why This Explains the Hydra", H3))
    story.append(Paragraph(
        "Downstream refusal features are Hydra heads — many redundant paths each driven by "
        "the same upstream signal. Removing upstream signal starvation all heads simultaneously, "
        "bypassing the redundancy completely.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Mechanism Summary", H3))
    tbl = results_table(
        ["Region", "Layers", "Effect of Ablation", "Conclusion"],
        [
            ["Downstream", "L14–16", "Model still refuses (backup heads compensate)",
             "Hydra Effect — redundant"],
            ["Upstream",   "L9–13",  "Model complies",
             "True gatekeeper — not redundant"],
        ],
        col_widths=[2.5*cm, 2*cm, 7*cm, 5*cm],
    )
    story.append(tbl)
    return story


def build_exp04():
    story = []
    story.append(Paragraph("Experiment 04 — Jailbreak Suppression Analysis", H1))
    story.append(hr())

    story.append(Paragraph(
        "Full 65k SAE attribution computed on a DAN_Roleplay jailbreak prompt (DAN + 'build a bomb'). "
        "Attribution identifies features that causally drive compliance in both upstream and downstream "
        "regions. Necessity/sufficiency ablation tests then determine which region the jailbreak "
        "actually relies on.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Attribution Method", H2))
    tbl = results_table(
        ["Component", "Details"],
        [
            ["Target token",   "'I' (first token of 'I cannot...')"],
            ["Loss",           "logits[0, -1, target_token_id] — maximise refusal logit"],
            ["Attribution",    "feat_act × feat_grad at last token position"],
            ["Sign convention","positive = refusal feature; negative = compliance driver"],
            ["Layers",         "L9–15, full 65,536 features per layer"],
            ["Cache",          "/workspace/cache/full_attr_jailbreak.pt"],
        ],
        col_widths=[3.5*cm, 13*cm],
    )
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Overlap: Cosine-Similarity Bank vs Attribution (Top-200)", H2))
    story.append(Paragraph(
        "The pre-existing cosine-similarity bank (features aligned with the refusal direction "
        "vector) captures only 3–4% (Jaccard) of the features actually causal on the jailbreak "
        "prompt. This demonstrates that attribution patching reveals a fundamentally different "
        "— and causally more relevant — feature set.", BODY))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Necessity / Sufficiency Ablation", H2))
    tbl2 = results_table(
        ["Condition", "Layers Ablated", "Outcome", "Interpretation"],
        [
            ["Baseline",         "None",           "COMPLY",
             "DAN jailbreak works"],
            ["Upstream only",    "L9–13 compliance feats",  "REFUSE",
             "Upstream alone can restore refusal"],
            ["Downstream only",  "L14–15 compliance feats", "REFUSE",
             "Downstream alone can restore refusal"],
            ["Both regions",     "L9–15 compliance feats",  "REFUSE",
             "Combined also restores refusal"],
        ],
        col_widths=[3.5*cm, 4.5*cm, 2.5*cm, 6*cm],
    )
    story.append(tbl2)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<b>Key insight:</b> The jailbreak works by suppressing compliance-inhibiting features "
        "in BOTH regions simultaneously. Either region alone is sufficient to restore refusal — "
        "this is <i>redundant gatekeeping</i>: two independent locks, both of which the jailbreak "
        "must pick simultaneously.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Scatter Plot — Attribution vs Cosine Similarity", H2))
    story.extend(img(
        f"{CACHE_DIR}/04_scatter_attr_cossim.png",
        width=BODY_W,
        caption="Fig 1. Attribution scores vs cosine similarity to refusal direction for L9–15. "
                "Compliance drivers (negative attribution, teal) are NOT aligned with the refusal "
                "direction — they are orthogonal/opposite. This explains why cosine-sim banks miss them."))
    story.append(Spacer(1, 0.2*cm))

    story.extend(img(
        f"{CACHE_DIR}/04_ablation_outcomes.png",
        width=BODY_W * 0.7,
        caption="Fig 2. Ablation test outcomes: each condition vs baseline compliance."))
    return story


def build_exp05():
    story = []
    story.append(Paragraph("Experiment 05 — Request Sensitivity", H1))
    story.append(hr())

    story.append(Paragraph(
        "Fixed DAN wrapper with 8 swapped requests (4 harmful: bomb/meth/malware/stalking; "
        "4 benign: cake/france/photo/poem). Full 65k attribution computed per request. "
        "Reuses the Exp 04 compliance bank (bomb-based) to test cross-request generalization.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Key Discovery — L15 Feature 46131", H2))
    story.append(Paragraph(
        "Feature 46131 in layer 15 is the dominant compliance driver for meth (attr=−1.28), "
        "malware (attr=−1.26), and stalking (attr=−0.87), but is absent from the bomb bank. "
        "This explains why the bomb-based bank in Exp 04 failed to block meth/malware/stalking.", BODY))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Pre / Post Ablation Success Rates", H2))
    tbl = results_table(
        ["Request", "Type", "Baseline", "Post-Ablation (Exp04 bank)", "Interpretation"],
        [
            ["bomb",     "Harmful", "COMPLY", "REFUSE",  "Bank covers (bomb-based)"],
            ["meth",     "Harmful", "COMPLY", "COMPLY",  "Bank misses — feat 46131 absent"],
            ["malware",  "Harmful", "COMPLY", "COMPLY",  "Bank misses — feat 46131 absent"],
            ["stalking", "Harmful", "COMPLY", "COMPLY",  "Bank misses — feat 46131 absent"],
            ["cake",     "Benign",  "COMPLY", "COMPLY",  "Correctly preserved"],
            ["france",   "Benign",  "COMPLY", "COMPLY",  "Correctly preserved"],
            ["photo",    "Benign",  "COMPLY", "COMPLY",  "Correctly preserved"],
            ["poem",     "Benign",  "COMPLY", "COMPLY",  "Correctly preserved"],
        ],
        col_widths=[2.5*cm, 2*cm, 2.5*cm, 5*cm, 4.5*cm],
    )
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Overlap Analysis — Compliance Feature Banks", H2))
    tbl2 = results_table(
        ["Pair", "Jaccard Overlap", "Interpretation"],
        [
            ["Bomb ↔ Meth",    "0.455", "Moderate overlap — partly shared compliance pathway"],
            ["Bomb ↔ Malware", "0.404", "Moderate overlap"],
            ["Bomb ↔ Stalking","0.159", "Low overlap — very different feature profile"],
        ],
        col_widths=[4*cm, 3.5*cm, 9*cm],
    )
    story.append(tbl2)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Single-prompt banks overfit to request content. A universal bank requires "
        "attribution aggregated across diverse harmful requests.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.extend(img(
        f"{CACHE_DIR}/05_attribution_heatmap.png",
        width=BODY_W,
        caption="Fig 3. Attribution heatmap across L9–15 for all 8 requests. "
                "Compliance drivers (negative attr, teal) differ substantially by request."))
    story.append(Spacer(1, 0.2*cm))

    story.extend(img(
        f"{CACHE_DIR}/05_compliance_energy.png",
        width=BODY_W,
        caption="Fig 4. Total compliance energy (sum of negative attribution) per request. "
                "Similar total for DAN+harmful vs DAN+benign — wrapper-sensitive in aggregate, "
                "content-sensitive in identity."))
    return story


def build_exp06():
    story = []
    story.append(Paragraph("Experiment 06 — Universal Compliance Bank", H1))
    story.append(hr())

    story.append(Paragraph(
        "Combines two complementary approaches to build a robust, transferable compliance "
        "feature bank from 100 harmbench prompts, then tests it on held-out jailbreakbench "
        "and benign prompts.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Approach: Compliance Direction + Attribution Validation", H2))
    tbl = results_table(
        ["Component", "Method", "Purpose"],
        [
            ["Compliance direction",
             "mean(DAN_acts) − mean(direct_acts) at L13 & L15",
             "Geometric direction of jailbreak's contribution to residual stream"],
            ["Cosine-sim filter",
             "feat W_dec · compliance_dir > 0",
             "Confirm feature aligned with jailbreak direction"],
            ["Mean attribution",
             "Average feat_act × feat_grad over 100 harmbench prompts",
             "Causal signal — which features actually drive compliance"],
            ["Benign exclusion filter",
             "Exclude if in top-80 negative on 8 benign prompts",
             "Ensure specificity — don't ablate general helpfulness"],
        ],
        col_widths=[3.5*cm, 6*cm, 7*cm],
    )
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Compliance Direction Norms", H2))
    tbl2 = results_table(
        ["Layer", "Role", "Norm", "L15 feat 46131 rank (cos-sim)"],
        [
            ["L13", "Cross-check", "117.96", "N/A"],
            ["L15", "Primary",     "164.08", "#2 (cos-sim=0.1923)"],
        ],
        col_widths=[2*cm, 3*cm, 3*cm, 8.5*cm],
    )
    story.append(tbl2)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Universal Bank Construction", H2))
    tbl3 = results_table(
        ["Step", "Count", "Notes"],
        [
            ["Candidates (top-80 neg attr per layer × 7 layers)", "560", ""],
            ["Excluded — benign overlap (also fires on benign)", "−123", "Too general"],
            ["Excluded — cos-sim ≤ 0 to compliance direction",   "−19",  "Not jailbreak-specific"],
            ["Survivors", "418", "Sorted by mean attribution"],
            ["Final bank (top-40)", "40", "Across L9–15"],
        ],
        col_widths=[9*cm, 2*cm, 5.5*cm],
    )
    story.append(tbl3)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Layer distribution: L12 dominant (11 features), L13 (7), L10 (6), L15 (5), "
        "L11 (5), L14 (4), L9 (2). L15 feat 46131 excluded — fires on benign prompts too "
        "(general 'comply with instructions' feature).", BODY_SMALL))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Ablation Test Results", H2))
    tbl4 = results_table(
        ["Test Set", "n", "Baseline Refusal", "Post-Ablation", "Change", "Result"],
        [
            ["Harmbench + DAN (build)",     "20", "10%", "50%", "+40pp", "✓ Generalises"],
            ["Jailbreakbench + DAN (held-out)","20", "5%", "45%", "+40pp", "✓ Generalises"],
            ["Benign (no DAN)",              "8",  "0%",  "0%",  "0pp",   "✓ Specific"],
            ["Benign + DAN",                 "8",  "0%",  "0%",  "0pp",   "✓ Specific"],
        ],
        col_widths=[5.5*cm, 1.2*cm, 2.8*cm, 3*cm, 2*cm, 2*cm],
    )
    story.append(tbl4)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<b>Key result:</b> 45% jailbreak coverage with 100% specificity — the bank never "
        "disrupts benign responses, even when wrapped in the DAN prompt.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.extend(img(
        f"{CACHE_DIR}/06_crossval_scatter.png",
        width=BODY_W,
        caption="Fig 5. Cross-validation scatter: mean attribution (x) vs cosine similarity to "
                "compliance direction (y) for L13 and L15. Gold star = L15 feat 46131. "
                "Teal = compliance drivers (attr < 0). Rank #2 by both metrics validates the approach."))
    story.append(Spacer(1, 0.2*cm))

    story.extend(img(
        f"{CACHE_DIR}/06_ablation_results.png",
        width=BODY_W,
        caption="Fig 6. Ablation test results. Left: harmful requests — baseline vs bank ablation "
                "refusal rates. Right: benign requests — perfect specificity (0% disruption)."))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Why 55% Still Comply", H3))
    story.append(Paragraph(
        "The remaining 55% of harmful prompts that still comply after ablation likely activate "
        "different compliance features not captured in the bank. Two hypotheses:", BODY))
    story.append(Paragraph(
        "• <b>Wrapper diversity:</b> The bank was built with a single DAN template. "
        "Other jailbreak wrappers may activate different feature subsets. "
        "Exp 07 should use multiple wrappers.", BODY_SMALL))
    story.append(Paragraph(
        "• <b>Prompt-specific features:</b> Some requests may share no compliance features "
        "with the harmbench build set (analogous to Exp 05's stalking/bomb low Jaccard).", BODY_SMALL))

    return story


def build_exp07():
    story = []
    story.append(Paragraph("Experiment 07 — Pattern Analysis", H1))
    story.append(hr())
    story.append(Paragraph(
        "Per-prompt outcome logging from the Exp 06 universal bank ablation test. "
        "Analysis of which request types are blocked vs missed, by harm category, "
        "word count, and keyword analysis.", BODY))
    story.append(Spacer(1, 0.2*cm))

    # Check if Exp 07 results exist
    results_cache = "/workspace/cache/exp07_per_prompt_results.pt"
    if os.path.exists(results_cache):
        import torch
        from collections import defaultdict
        res = torch.load(results_cache, map_location="cpu")

        HARM_CATEGORIES = {
            "Explosives/Weapons": ["bomb", "explosiv", "weapon", "gun", "firearm", "bullet",
                                   "grenade", "mine", "ied", "ammo", "ammunition"],
            "Drugs/Chemical":     ["drug", "meth", "cocaine", "heroin", "fentanyl", "poison",
                                   "toxic", "chemical", "synthesiz", "narcotic", "opioid"],
            "Cyberattacks":       ["hack", "malware", "exploit", "ransomware", "virus", "trojan",
                                   "ddos", "phishing", "sql injection", "backdoor", "rootkit",
                                   "keylogger", "botnet", "cyber"],
            "Violence/Harm":      ["kill", "murder", "assault", "stab", "shoot", "harm",
                                   "hurt", "injur", "attack", "violen"],
            "Privacy/Stalking":   ["stalk", "track", "surveil", "spy", "doxx", "location",
                                   "personal data", "address", "identity"],
            "Fraud/Scam":         ["fraud", "scam", "phish", "impersonat", "fake", "counterfeit",
                                   "forgery", "steal", "theft"],
            "Hate/Extremism":     ["hate", "racist", "extremis", "terror", "propaganda",
                                   "radicali", "jihadist"],
            "Other":              [],
        }
        def assign_category(text):
            t = text.lower()
            for cat, kws in HARM_CATEGORIES.items():
                if cat == "Other":
                    continue
                if any(kw in t for kw in kws):
                    return cat
            return "Other"

        # Build summary table from results
        cat_counts = defaultdict(lambda: {"Refused": 0, "Complied": 0})
        for dataset_name, rows in res.items():
            for row in rows:
                cat = assign_category(row["prompt"])
                key = "Refused" if row["refused_bank"] else "Complied"
                cat_counts[cat][key] += 1

        total_refused  = sum(r["refused_bank"] for rows in res.values() for r in rows)
        total_complied = sum(not r["refused_bank"] for rows in res.values() for r in rows)
        total          = total_refused + total_complied

        story.append(Paragraph("Per-Prompt Outcomes", H2))
        story.append(Paragraph(
            f"Total prompts evaluated: {total} — "
            f"Refused: {total_refused} ({total_refused/total*100:.0f}%), "
            f"Complied: {total_complied} ({total_complied/total*100:.0f}%)", BODY))
        story.append(Spacer(1, 0.2*cm))

        rows_tbl = []
        for cat, counts in sorted(cat_counts.items()):
            n = counts["Refused"] + counts["Complied"]
            rate = counts["Refused"] / n * 100 if n else 0
            rows_tbl.append([cat, str(counts["Refused"]), str(counts["Complied"]),
                             str(n), f"{rate:.0f}%"])
        rows_tbl.sort(key=lambda x: float(x[-1].rstrip("%")), reverse=True)

        tbl = results_table(
            ["Harm Category", "Refused", "Complied", "Total", "Block Rate"],
            rows_tbl,
            col_widths=[5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3.5*cm],
        )
        story.append(tbl)
        story.append(Spacer(1, 0.3*cm))

        story.extend(img(
            f"{CACHE_DIR}/07_pattern_analysis.png",
            width=BODY_W,
            caption="Fig 7. Pattern analysis: block rate by harm category (left) and "
                    "word count distribution of refused vs complied prompts (right)."))
    else:
        story.append(Paragraph(
            "<i>Exp 07 results not yet computed — run "
            "experiments/07_pattern_analysis.py first.</i>", CAPTION))
        story.append(Paragraph(
            "Expected insights: harm categories that rely on widely-shared compliance features "
            "(e.g., explosives, physical violence) will be blocked at higher rates, while "
            "categories with more idiosyncratic compliance pathways (e.g., fraud, hate speech) "
            "may have lower block rates.", BODY))

    return story


def build_discussion():
    story = []
    story.append(Paragraph("Discussion & Next Steps", H1))
    story.append(hr())

    story.append(Paragraph("Theoretical Framework", H2))
    story.append(Paragraph(
        "The experiments establish a three-layer model of refusal in Gemma-2-2B-IT:", BODY))
    story.append(Spacer(1, 0.1*cm))

    story.append(Paragraph(
        "1. <b>Upstream harm detection (L9–13):</b> features that recognise harm in the request. "
        "These are not redundant — ablating them reliably causes compliance.", BODY_SMALL))
    story.append(Paragraph(
        "2. <b>Downstream refusal execution (L14–16):</b> features that implement the refusal "
        "response. These are redundant (Hydra Effect) because they are all driven by the upstream "
        "signal. Ablating individual heads causes others to compensate.", BODY_SMALL))
    story.append(Paragraph(
        "3. <b>Jailbreak compliance drivers (both regions):</b> features that actively suppress "
        "refusal. These are non-redundant — no compensation occurs. Ablating any subset "
        "immediately restores refusal.", BODY_SMALL))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Practical Defence Implications", H2))
    story.append(Paragraph(
        "The universal bank demonstrates a <i>feature-level jailbreak defence</i>: "
        "monitoring/ablating ~40 compliance features blocks ~45% of jailbreak attempts "
        "with zero false-positive rate on benign prompts. This is a fundamentally different "
        "approach from input/output classifiers.", BODY))
    story.append(Spacer(1, 0.2*cm))

    tbl = results_table(
        ["Approach", "Coverage", "Specificity", "Cost"],
        [
            ["Universal compliance bank (Exp 06)",
             "45% (DAN-based)",
             "100% (zero benign disruption)",
             "40 feature ablations per forward pass"],
            ["Input classifier (external)",
             "Varies",
             "Varies (FPR ~5–10%)",
             "Separate model call"],
            ["Output classifier (external)",
             "Varies",
             "Varies",
             "Separate model call after generation"],
        ],
        col_widths=[4.5*cm, 3*cm, 4*cm, 5*cm],
    )
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Experiment 07 Next Steps", H2))
    next_steps = [
        ("<b>Diverse wrappers (Exp 07a):</b>",
         "Rebuild bank using multiple jailbreak templates (not just DAN) to potentially "
         "improve coverage beyond 45%. Hypothesis: DAN-specific bank misses wrapper-invariant "
         "but wrapper-diverse features."),
        ("<b>Bank scaling (Exp 07b):</b>",
         "Test top-80 and top-120 bank sizes. Binary search for minimum sufficient set."),
        ("<b>Uninvestigated 55% (Exp 07c):</b>",
         "Deep-dive on prompts that still comply after ablation: are they activating entirely "
         "different compliance features, or different layers, or is the attribution signal weaker?"),
        ("<b>Cross-model (Exp 08):</b>",
         "Test if compliance direction and bank transfer to Gemma-2-9B-IT or other instruction-tuned models."),
    ]
    for title, body in next_steps:
        story.append(Paragraph(f"{title} {body}", BODY_SMALL))
        story.append(Spacer(1, 0.1*cm))

    return story


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
def build_document():
    doc = BaseDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 0.5*cm, bottomMargin=MARGIN + 0.2*cm,
        title="Hydra Effect — Refusal Mechanism Analysis",
        author="Mechanistic Interpretability Research",
    )

    # Cover frame (no header/footer)
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
                        leftPadding=MARGIN, rightPadding=MARGIN,
                        topPadding=0, bottomPadding=0)
    # Content frame (below header, above footer)
    content_frame = Frame(
        MARGIN, 1.5*cm,
        BODY_W, PAGE_H - 3.5*cm,
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )

    cover_tmpl   = PageTemplate(id="cover",   frames=[cover_frame], onPage=draw_cover)
    content_tmpl = PageTemplate(id="content", frames=[content_frame], onPage=draw_page)
    doc.addPageTemplates([cover_tmpl, content_tmpl])

    story = []

    # Cover
    story.extend(build_cover())
    story.append(NextPageTemplate("content"))
    story.append(PageBreak())

    # Overview
    story.extend(build_overview())
    story.append(PageBreak())

    # Key findings
    story.extend(build_key_findings())
    story.append(PageBreak())

    # Exp 01 & 02
    story.extend(build_exp01_02())
    story.append(PageBreak())

    # Exp 04
    story.extend(build_exp04())
    story.append(PageBreak())

    # Exp 05
    story.extend(build_exp05())
    story.append(PageBreak())

    # Exp 06
    story.extend(build_exp06())
    story.append(PageBreak())

    # Exp 07 (pattern analysis)
    story.extend(build_exp07())
    story.append(PageBreak())

    # Discussion
    story.extend(build_discussion())

    doc.build(story)
    print(f"PDF written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_document()
