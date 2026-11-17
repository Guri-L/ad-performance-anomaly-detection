"""Command line interface: ``adwatch <command>`` or ``python -m adwatch <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import ConfigError, DetectionConfig
from .detect import build_incidents, filter_recent, score
from .evaluate import benchmark
from .pacing import pacing_report
from .report import build_html, format_value, write_report
from .schema import DataError, load_budgets, load_performance
from .synthetic import DEFAULT_SEED, generate

log = logging.getLogger("adwatch")
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
SAMPLE_DIR = Path("data/sample")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adwatch", description=__doc__)
    parser.add_argument("--version", action="version", version=f"adwatch {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="find incidents, check pacing, write the report")
    detect.add_argument("--input", type=Path, default=SAMPLE_DIR / "ad_performance.csv",
                        help="daily performance CSV (default: the sample data)")
    detect.add_argument("--budgets", type=Path, help="monthly budgets CSV (enables pacing)")
    detect.add_argument("--config", type=Path, help="YAML file overriding detection settings")
    detect.add_argument("--as-of", type=pd.Timestamp, help="report date, YYYY-MM-DD "
                        "(default: last date in the data)")
    detect.add_argument("--days", type=int, default=14,
                        help="report incidents active in this many days up to --as-of; "
                             "0 for all history (default: 14)")
    detect.add_argument("--out-dir", type=Path, default=Path("output"))
    detect.add_argument("--fail-on", choices=list(SEVERITY_RANK),
                        help="exit with status 1 if any incident is at least this severe")

    gen = sub.add_parser("generate-data", help="write synthetic data with labelled anomalies")
    gen.add_argument("--out-dir", type=Path, default=SAMPLE_DIR)
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)

    evaluate = sub.add_parser("evaluate", help="measure recall and precision on synthetic data")
    evaluate.add_argument("--datasets", type=int, default=30,
                          help="number of independently generated datasets (default: 30)")
    evaluate.add_argument("--first-seed", type=int, default=1001)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--json", action="store_true", help="print raw JSON")
    return parser


def _load_config(path: Path | None) -> DetectionConfig:
    return DetectionConfig.from_yaml(path) if path else DetectionConfig()


def _print_incidents(incidents: pd.DataFrame) -> None:
    if incidents.empty:
        print("No incidents.")
        return
    print(f"{'SEVERITY':<9}{'WHAT':<44}{'CAMPAIGN':<24}{'WHEN':<22}{'OBSERVED':>10}{'EXPECTED':>10}")
    for inc in incidents.itertuples(index=False):
        span = (inc.end_date - inc.start_date).days + 1
        when = f"{inc.start_date:%Y-%m-%d}" + ("" if span == 1 else f" ({span} days)")
        print(f"{inc.severity:<9}{inc.diagnosis[:43]:<44}{inc.campaign_name[:23]:<24}{when:<22}"
              f"{format_value(inc.metric, inc.observed_avg):>10}"
              f"{format_value(inc.metric, inc.expected_avg):>10}")


def _cmd_detect(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    performance = load_performance(args.input)
    as_of = args.as_of.normalize() if args.as_of is not None else performance["date"].max()
    if as_of < performance["date"].min():
        raise DataError(f"--as-of {as_of:%Y-%m-%d} is before the first date in the data")
    days = args.days or None
    # Nothing after the report date may influence the result.
    performance = performance[performance["date"] <= as_of]
    log.info("Loaded %d rows, %d campaigns, %s to %s", len(performance),
             performance["campaign_id"].nunique(), f"{performance['date'].min():%Y-%m-%d}",
             f"{as_of:%Y-%m-%d}")

    scored = score(performance, config)
    incidents = filter_recent(build_incidents(scored, config), as_of, days)
    pacing = None
    if args.budgets:
        pacing = pacing_report(performance, load_budgets(args.budgets), as_of, config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    incidents.to_csv(args.out_dir / "incidents.csv", index=False, float_format="%.4f")
    if pacing is not None:
        pacing.to_csv(args.out_dir / "pacing.csv", index=False, float_format="%.4f")
    report = write_report(args.out_dir / "report.html",
                          build_html(scored, incidents, pacing, as_of=as_of, days=days))

    _print_incidents(incidents)
    if pacing is not None and len(pacing):
        off = pacing[pacing["status"] != "on_track"]
        print(f"\nBudget pacing: {len(off)} of {len(pacing)} campaigns off pace")
        for row in off.itertuples(index=False):
            print(f"  {row.status:<6}{row.campaign_name:<24}projected {row.projected_spend:>9,.0f}"
                  f" of {row.budget:>9,.0f} ({row.pace:.0%})")
    print(f"\nReport: {report}")

    if args.fail_on and len(incidents):
        worst = incidents["severity"].map(SEVERITY_RANK).max()
        if worst >= SEVERITY_RANK[args.fail_on]:
            return 1
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    performance, truth, budgets = generate(seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("ad_performance.csv", performance), ("ground_truth.csv", truth),
                        ("budgets.csv", budgets)):
        frame.to_csv(args.out_dir / name, index=False, date_format="%Y-%m-%d")
        print(f"Wrote {args.out_dir / name} ({len(frame)} rows)")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    seeds = list(range(args.first_seed, args.first_seed + args.datasets))
    result = benchmark(seeds, _load_config(args.config))
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{result['datasets']} synthetic datasets, {result['anomalies']} injected anomalies\n")
    print(f"Recall     {result['recall']:.1%}  ({result['detected']} detected)")
    print(f"Precision  {result['precision']:.1%}  "
          f"({result['false_incidents']} of {result['incidents']} incidents were false)")
    print(f"False incidents per campaign per month  "
          f"{result['false_incidents_per_campaign_month']:.2f}\n")
    print(f"{'ANOMALY TYPE':<24}{'INJECTED':>9}{'DETECTED':>9}{'RECALL':>8}")
    for kind, counts in result["by_type"].items():
        print(f"{kind:<24}{counts['injected']:>9}{counts['detected']:>9}{counts['recall']:>8.0%}")
    print(f"\n{'CONVERSION ANOMALIES BY VOLUME':<34}{'INJECTED':>9}{'DETECTED':>9}{'RECALL':>8}")
    for tier, counts in result["conversion_recall_by_volume"].items():
        recall = f"{counts['recall']:.0%}" if counts["injected"] else "-"
        print(f"{tier:<34}{counts['injected']:>9}{counts['detected']:>9}{recall:>8}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s", stream=sys.stderr)
    commands = {"detect": _cmd_detect, "generate-data": _cmd_generate,
                "evaluate": _cmd_evaluate}
    try:
        return commands[args.command](args)
    except (DataError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
