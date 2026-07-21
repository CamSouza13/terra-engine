"""PDF system reports.

Builds a printable report for a workspace: operating summary, per-channel
statistics over the stored history, the calibrated parameters in force, recent
alert events, and the node roster. Uses reportlab, an optional extra
(``pip install terra-engine[reports]``); the server returns a hint if it is
missing. The caller assembles a plain ``ctx`` dict (see ``build_pdf``), so this
module has no coupling to the server or database.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle)
    HAS_REPORTLAB = True
except Exception:  # pragma: no cover - exercised only without the extra
    HAS_REPORTLAB = False

INK = "#1c1c1e"
SEC = "#6e6e73"
LINE = "#d8d8d4"


def _fmt_ts(ts):
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def channel_stats(history: list, channels: list) -> list:
    """Min / mean / max / last per channel over the history records."""
    rows = []
    for ch in channels:
        vals = [r.get("channels", {}).get(ch) for r in history]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            rows.append([ch, "—", "—", "—", "—", "0"])
            continue
        rows.append([ch, f"{min(vals):.3g}", f"{sum(vals) / len(vals):.3g}",
                     f"{max(vals):.3g}", f"{vals[-1]:.3g}", str(len(vals))])
    return rows


def build_pdf(ctx: dict) -> bytes:
    """Render a report PDF from a context dict and return the bytes.

    ctx keys: workspace, domain, source, cycles, calibrated (dict), nodes (list of
    {name,domain,last_seen,stale}), alerts (list of {created,message,delivered}),
    history (list of records), channels (list of names).
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("PDF reports need the optional extra: "
                           "pip install terra-engine[reports]")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.7 * inch,
                            bottomMargin=0.7 * inch, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, title="Terra System Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=colors.HexColor(INK), spaceAfter=2)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9.5, textColor=colors.HexColor(SEC))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor(INK), spaceBefore=16, spaceAfter=6)
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8.5, textColor=colors.HexColor(SEC))

    def tbl(data, widths):
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(SEC)),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(LINE)),
            ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor(LINE)),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    story = []
    story.append(Paragraph("Terra — System Report", h1))
    story.append(Paragraph(f"Workspace: {ctx.get('workspace') or 'local'} · Generated {_fmt_ts(datetime.now(timezone.utc).timestamp())}", sub))
    story.append(Spacer(1, 10))

    cal = ctx.get("calibrated") or {}
    summary = [
        ["Field", "Value"],
        ["Domain", str(ctx.get("domain", "—"))],
        ["Data source", str(ctx.get("source", "—"))],
        ["Engine cycles", str(ctx.get("cycles", 0))],
        ["Calibration", "calibrated (" + ", ".join(cal.keys()) + ")" if cal else "defaults"],
        ["Nodes", f"{sum(1 for n in ctx.get('nodes', []) if not n.get('stale'))} live / {len(ctx.get('nodes', []))} total"],
        ["History points", str(len(ctx.get("history", [])))],
    ]
    story.append(Paragraph("Operating summary", h2))
    story.append(tbl(summary, [2.0 * inch, 4.0 * inch]))

    story.append(Paragraph("Channel statistics", h2))
    cs = channel_stats(ctx.get("history", []), ctx.get("channels", []))
    story.append(tbl([["Channel", "Min", "Mean", "Max", "Last", "N"]] + cs,
                     [1.7 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch]))

    if cal:
        story.append(Paragraph("Calibrated parameters", h2))
        story.append(tbl([["Parameter", "Value"]] + [[k, f"{v:.4g}" if isinstance(v, (int, float)) else str(v)] for k, v in cal.items()],
                         [3.0 * inch, 3.0 * inch]))

    story.append(Paragraph("Recent alerts", h2))
    al = ctx.get("alerts", [])
    if al:
        rows = [["Time", "Event", "Delivered"]]
        for e in al[:20]:
            rows.append([_fmt_ts(e.get("created")), str(e.get("message", ""))[:70],
                         "yes" if e.get("delivered") else "stored"])
        story.append(tbl(rows, [1.6 * inch, 3.8 * inch, 0.9 * inch]))
    else:
        story.append(Paragraph("No alert events in the record.", small))

    story.append(Paragraph("Nodes", h2))
    nodes = ctx.get("nodes", [])
    if nodes:
        rows = [["Node", "Domain", "Last seen", "Status"]]
        for n in nodes:
            rows.append([str(n.get("name", "—")), str(n.get("domain") or "—"),
                         _fmt_ts(n.get("last_seen")), "stale" if n.get("stale") else "live"])
        story.append(tbl(rows, [2.0 * inch, 1.4 * inch, 1.9 * inch, 0.9 * inch]))
    else:
        story.append(Paragraph("No nodes enrolled.", small))

    story.append(Spacer(1, 18))
    story.append(Paragraph("Generated by Terra · terralaboratories.com", small))

    doc.build(story)
    return buf.getvalue()
