"""Charts and the self-contained HTML report."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .detect import MISSING

# Colour roles. Text never takes a series colour; flagged days use the reserved
# "critical" status colour together with a marker shape and a text label.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
BAND = "#cde2fb"
CRITICAL = "#d03b3b"

METRIC_LABELS = {
    "spend": "Spend",
    "impressions": "Impressions",
    "cpc": "Cost per click",
    "ctr": "Click-through rate",
    "conversion_rate": "Conversion rate",
    MISSING: "Data",
}
PERCENT_METRICS = {"ctr", "conversion_rate"}
CONTEXT_DAYS = 35


def format_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "-"
    if metric in PERCENT_METRICS:
        return f"{value:.2%}"
    if metric == "impressions":
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def incident_chart(scored: pd.DataFrame, incident: pd.Series) -> Figure:
    """Observed values against the expected range around one incident."""
    metric = incident["metric"]
    series = scored[
        (scored["campaign_id"] == incident["campaign_id"]) & (scored["metric"] == metric)
    ].sort_values("date")
    start = incident["start_date"] - pd.Timedelta(days=CONTEXT_DAYS)
    end = incident["end_date"] + pd.Timedelta(days=7)
    series = series[(series["date"] >= start) & (series["date"] <= end)]
    flagged = series[series["is_anomaly"] & (series["direction"] == incident["direction"])
                     & (series["date"] >= incident["start_date"])
                     & (series["date"] <= incident["end_date"])]

    fig, ax = plt.subplots(figsize=(8.4, 3.3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(series["date"], series["lower"], series["upper"], color=BAND,
                    linewidth=0, label="Expected range")
    ax.plot(series["date"], series["expected"], color=TEXT_MUTED, linewidth=1.2,
            linestyle=(0, (4, 3)), label="Expected")
    ax.plot(series["date"], series["observed"], color=SERIES, linewidth=2, label="Observed")
    ax.scatter(flagged["date"], flagged["observed"], s=70, color=CRITICAL, marker="D",
               edgecolors=SURFACE, linewidths=2, zorder=3, clip_on=False, label="Flagged day")

    if not flagged.empty:
        # The callout sits in the title row, where it cannot collide with the line.
        worst = flagged.loc[flagged["z_score"].abs().idxmax()]
        ax.text(1.0, 1.035,
                f"{worst['date']:%d %b}: {format_value(metric, worst['observed'])} "
                f"(expected {format_value(metric, worst['expected'])})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color=TEXT_SECONDARY)

    ax.set_title(f"{incident['campaign_name']}: {METRIC_LABELS[metric].lower()} per day",
                 loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
    ax.yaxis.set_major_formatter(lambda v, _: format_value(metric, v))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5, length=0)
    legend = ax.legend(loc="upper left", ncols=4, frameon=False, fontsize=8.5,
                       bbox_to_anchor=(0, -0.14))
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    return fig


def figure_to_png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


def _table(headers: list[str], rows: list[list[str]], numeric: set[int]) -> str:
    head = "".join(
        f'<th class="{"num" if i in numeric else ""}">{html.escape(h)}</th>'
        for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if i in numeric else ""}">{cell}</td>'
            for i, cell in enumerate(row)) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _badge(kind: str, label: str) -> str:
    return f'<span class="badge {html.escape(kind)}">{html.escape(label)}</span>'


def _date_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start:%d %b}" if start == end else f"{start:%d %b} to {end:%d %b}"


def _incident_rows(incidents: pd.DataFrame) -> list[list[str]]:
    rows = []
    for inc in incidents.itertuples(index=False):
        metric = inc.metric
        change = "-" if pd.isna(inc.pct_change) else f"{inc.pct_change:+.0%}"
        impact = "-" if pd.isna(inc.estimated_impact) else f"{inc.estimated_impact:+,.0f}"
        rows.append([
            _badge(inc.severity, inc.severity.capitalize()),
            html.escape(inc.diagnosis),
            html.escape(f"{inc.campaign_name} ({inc.platform})"),
            _date_range(inc.start_date, inc.end_date),
            format_value(metric, inc.observed_avg),
            format_value(metric, inc.expected_avg),
            change,
            impact,
        ])
    return rows


def _pacing_rows(pacing: pd.DataFrame) -> list[list[str]]:
    labels = {"over": "Over budget", "under": "Under budget", "on_track": "On track"}
    rows = []
    for row in pacing.itertuples(index=False):
        runs_out = "-" if pd.isna(row.budget_runs_out) else f"{row.budget_runs_out:%d %b}"
        rows.append([
            _badge(row.status, labels[row.status]),
            html.escape(f"{row.campaign_name} ({row.platform})"),
            f"{row.budget:,.0f}",
            f"{row.spend_to_date:,.0f}",
            f"{row.projected_spend:,.0f}",
            f"{row.pace:.0%}",
            f"{row.projected_variance:+,.0f}",
            runs_out,
        ])
    return rows


CSS = """
body{margin:0;background:#f4f3ef;color:#0b0b0b;font:15px/1.5 system-ui,-apple-system,"Segoe UI",
Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 12px}
.sub{color:#52514e;margin:0 0 24px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:8px;padding:14px 16px}
.tile .n{font-size:28px;font-weight:600;line-height:1.1}.tile .l{color:#52514e;font-size:13px}
.card{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:8px;padding:8px 16px;
overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;color:#52514e;font-weight:600;font-size:12.5px;padding:10px 10px;
border-bottom:1px solid #c3c2b7;white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid #e1e0d9;vertical-align:top}
tr:last-child td{border-bottom:0}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12.5px;font-weight:600;
border:1px solid #c3c2b7;color:#0b0b0b;white-space:nowrap}
.badge::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;
margin-right:6px;background:#898781}
.badge.high::before,.badge.over::before{background:#d03b3b}
.badge.medium::before,.badge.under::before{background:#ec835a}
.badge.low::before{background:#fab219}.badge.on_track::before{background:#0ca30c}
figure{margin:0 0 16px;background:#fcfcfb;border:1px solid #e1e0d9;border-radius:8px;
padding:12px}
figure img{width:100%;height:auto;display:block}
figcaption{color:#52514e;font-size:13.5px;padding:6px 4px 2px}
.empty{color:#52514e;padding:14px 4px}
footer{color:#898781;font-size:12.5px;margin-top:40px}
"""


def build_html(
    scored: pd.DataFrame,
    incidents: pd.DataFrame,
    pacing: pd.DataFrame | None,
    *,
    as_of: pd.Timestamp,
    days: int | None,
    max_charts: int = 6,
) -> str:
    period = f"the {days} days to {as_of:%d %B %Y}" if days else f"all data to {as_of:%d %B %Y}"
    counts = incidents["severity"].value_counts() if len(incidents) else pd.Series(dtype=int)
    tiles = [
        (len(incidents), "incidents"),
        (int(counts.get("high", 0)), "high severity"),
        (incidents["campaign_id"].nunique() if len(incidents) else 0, "campaigns affected"),
    ]
    if pacing is not None and len(pacing):
        tiles.append((int((pacing["status"] != "on_track").sum()), "campaigns off budget pace"))
    tile_html = "".join(
        f'<div class="tile"><div class="n">{n}</div><div class="l">{html.escape(label)}</div></div>'
        for n, label in tiles)

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Ad performance report, {as_of:%d %b %Y}</title><style>{CSS}</style></head>",
        "<body><main>",
        "<h1>Ad performance anomaly report</h1>",
        f"<p class='sub'>Incidents active in {html.escape(period)}</p>",
        f"<div class='tiles'>{tile_html}</div>",
        "<h2>Incidents</h2>",
    ]
    if len(incidents):
        parts.append("<div class='card'>" + _table(
            ["Severity", "What happened", "Campaign", "When", "Observed", "Expected", "Change",
             "Est. impact"],
            _incident_rows(incidents), numeric={4, 5, 6, 7}) + "</div>")
    else:
        parts.append("<div class='card'><p class='empty'>No incidents in this period.</p></div>")

    charted = incidents[incidents["metric"] != MISSING].head(max_charts)
    if len(charted):
        parts.append("<h2>Largest incidents</h2>")
    for _, incident in charted.iterrows():
        png = base64.b64encode(figure_to_png(incident_chart(scored, incident))).decode("ascii")
        caption = (f"{incident['diagnosis']}, "
                   f"{_date_range(incident['start_date'], incident['end_date'])}: "
                   f"average {format_value(incident['metric'], incident['observed_avg'])} against "
                   f"{format_value(incident['metric'], incident['expected_avg'])} expected.")
        alt = f"Line chart of {METRIC_LABELS[incident['metric']].lower()} for " \
              f"{incident['campaign_name']} with flagged days marked"
        parts.append(
            f"<figure><img alt='{html.escape(alt, quote=True)}' "
            f"src='data:image/png;base64,{png}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")

    if pacing is not None:
        parts.append(f"<h2>Budget pacing, {as_of:%B %Y}</h2>")
        if len(pacing):
            parts.append("<div class='card'>" + _table(
                ["Status", "Campaign", "Budget", "Spent", "Projected", "Pace", "Variance",
                 "Budget runs out"],
                _pacing_rows(pacing), numeric={2, 3, 4, 5, 6, 7}) + "</div>")
        else:
            parts.append("<div class='card'><p class='empty'>No budgets for this month.</p></div>")

    parts.append("<footer>Estimated impact is in the account currency: extra or missing spend "
                 "for spend and CPC incidents, revenue for conversion-rate incidents.</footer>")
    parts.append("</main></body></html>")
    return "".join(parts)


def write_report(path: str | Path, document: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
