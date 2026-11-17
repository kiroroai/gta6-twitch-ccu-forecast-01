#!/usr/bin/env python3
"""Forecast launch-day Twitch viewers hour by hour (make.fun program/v1).

Reads one case file per run (schema makefun/twitch-ccu-case/v1) and writes,
for hours 2..24, five quantiles of that hour's concurrent-viewer count.

Method (computed fresh from the case's own comparables at run time):
  1. For every past launch in `comparables`, align its hourly viewer counts
     to its release day and take log ratios ln(viewers_h / viewers_1).
  2. Keep launches that resemble a big console rolling-midnight launch:
     tier "comparable", a solid hour-1 print, most hours present. Launches
     whose notes mention early access, a simultaneous global unlock, a PC
     unlock, free-to-play or a network test are dropped from the fit,
     because their day-one shape is driven by that event rather than by
     the rolling unlock; they return at half weight only if too few clean
     launches survive.
  3. Big launches grow less over the day than small ones (their hour 1
     already holds a huge audience), so fit one pooled size slope: each
     hour's log ratio = alpha_h + beta * (ln h1_launch - ln h1_case),
     with per-hour intercepts alpha_h and a single shared beta, shrunk
     and clamped so it never extrapolates wildly.
  4. Smooth the alpha_h curve lightly across neighbouring hours, then
     spread the quantiles with the pooled empirical residual quantiles,
     scaled per hour by that hour's residual spread. The median is the
     conditional median; the bands are honest about how much launch
     curves actually disagree.
  5. Everything is anchored on the case's own hour-1 print, so the level
     follows the launch in front of us while the shape comes from history.

Only the standard library is used. No network, no randomness, no clock:
two runs on the same case write identical bytes.
"""

import json
import math
import os
import re
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Phrases in a comparable's unlock note that mark its release-day shape as
# driven by something other than a rolling midnight console launch.
SUSPECT_MARKERS = (
    "early access",
    "simultaneous",
    "single global",
    "f2p",
    "free-to-play",
    "network test",
    "steam global",
    "pc unlock",
    "pc single",
    "multiplayer beta",
    "beta weekend",
)

TIER_RE = re.compile(r"Tier:\s*([a-z][a-z\-]*)")


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_index(day_start, when):
    """1-based hour of the release day that `when` falls in, or None."""
    delta = (parse_time(when) - day_start).total_seconds()
    h = int(delta // 3600) + 1
    return h if 1 <= h <= 24 else None


def comparable_curves(case, strict=True):
    """Return [(log_h1, weight, {hour: log_ratio})] for usable comparables.

    Strict pass: drop any launch whose unlock note marks its release-day
    shape as driven by something other than a rolling midnight console
    launch (early access, a simultaneous global unlock, a PC unlock,
    free-to-play, a network test). If too few survive, fall back to
    keeping them at half weight rather than trusting nothing.
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        tier = None
        unlock = comp.get("unlock") or ""
        match = TIER_RE.search(unlock)
        if match:
            tier = match.group(1)
        if tier in ("weak", "early-access-contaminated", "target-history"):
            continue
        lowered = unlock.lower()
        suspect = any(m in lowered for m in SUSPECT_MARKERS)
        if strict and suspect:
            continue
        weight = 0.5 if suspect else 1.0
        by_hour = {}
        for point in comp.get("hourly") or []:
            viewers = point.get("viewers")
            if not viewers or viewers <= 0:
                continue
            h = hour_index(day_start, point["from"])
            if h is None:
                continue
            # Average if a source ever gives two points in one hour.
            if h in by_hour:
                by_hour[h] = (by_hour[h] + float(viewers)) / 2.0
            else:
                by_hour[h] = float(viewers)
        first = by_hour.get(1)
        present = sum(1 for h in range(2, 25) if h in by_hour)
        # A tiny hour-1 print means the category was barely sampled before
        # the unlock, so its ratios explode and say nothing about a big
        # launch; require a real audience and most hours present.
        if first is None or first < 20000 or present < 15:
            continue
        log_first = math.log(first)
        ratios = {
            h: math.log(v / first) for h, v in by_hour.items() if h >= 2
        }
        curves.append((log_first, weight, ratios))
    if strict and len(curves) < 5:
        return comparable_curves(case, strict=False)
    return curves


def weighted_quantile(pairs, level):
    """Weighted quantile of [(value, weight)] by linear interpolation."""
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = level * total
    acc = 0.0
    for i, (value, weight) in enumerate(ordered):
        prev = acc
        acc += weight
        if acc >= target:
            if i == 0 or weight <= 0:
                return value
            frac = (target - prev) / weight
            return ordered[i - 1][0] * (1.0 - frac) + value * frac
    return ordered[-1][0]


def fit_shape(curves, log_p):
    """Fit alpha_h + beta * (ln h1 - ln p); return (median_log, offsets).

    median_log[h] is the forecast median log ratio for hour h and offsets
    maps each level to its log offset from that median.
    """
    hours = list(range(2, 25))
    # Pooled size slope beta from within-hour demeaned log ratios.
    num = 0.0
    den = 0.0
    for h in hours:
        rows = [
            (log_first - log_p, ratio, weight)
            for log_first, weight, ratios in curves
            if h in ratios
            for ratio in [ratios[h]]
        ]
        if not rows:
            continue
        wsum = sum(w for _, _, w in rows)
        mean = sum(r * w for _, r, w in rows) / wsum
        for x, r, w in rows:
            num += w * x * (r - mean)
            den += w * x * x
    beta = num / den if den > 1e-9 else 0.0
    beta = max(-0.45, min(0.05, 0.8 * beta))

    # Per-hour intercepts, then a light 3-point smoothing across hours.
    alpha = {}
    residuals = []
    for h in hours:
        rows = [
            (ratio - beta * (log_first - log_p), weight)
            for log_first, weight, ratios in curves
            if h in ratios
            for ratio in [ratios[h]]
        ]
        if rows:
            wsum = sum(w for _, w in rows)
            alpha[h] = sum(v * w for v, w in rows) / wsum
        else:
            alpha[h] = 0.0
    smoothed = {}
    for i, h in enumerate(hours):
        neighbours = []
        if i > 0:
            neighbours.append(alpha[hours[i - 1]])
        neighbours.append(alpha[h])
        if i + 1 < len(hours):
            neighbours.append(alpha[hours[i + 1]])
        weights = [0.25, 0.5, 0.25][: len(neighbours)]
        if len(neighbours) == 2:
            weights = [0.5, 0.5] if i == 0 else [0.25, 0.75]
        total = sum(weights)
        smoothed[h] = sum(v * w for v, w in zip(neighbours, weights)) / total
    # Residuals against the smoothed curve, pooled over hours.
    for log_first, weight, ratios in curves:
        x = log_first - log_p
        for h, ratio in ratios.items():
            residuals.append((ratio - (smoothed[h] + beta * x), weight))
    pooled_sd = math.sqrt(
        sum(w * v * v for v, w in residuals)
        / max(sum(w for _, w in residuals), 1e-9)
    )
    if pooled_sd < 1e-6:
        pooled_sd = 0.3
    resid_q = {
        tau: weighted_quantile(residuals, tau) for tau in LEVELS
    }
    median_shift = resid_q[0.5]

    median_log = {}
    offsets = {}
    for h in hours:
        rows = [
            (ratio - (smoothed[h] + beta * (log_first - log_p)), weight)
            for log_first, weight, ratios in curves
            if h in ratios
            for ratio in [ratios[h]]
        ]
        if rows:
            var = sum(w * v * v for v, w in rows) / sum(w for _, w in rows)
            hour_sd = math.sqrt(max(var, 0.0))
        else:
            hour_sd = pooled_sd
        # Shrink the hour's spread halfway toward the pooled spread.
        hour_sd = math.sqrt(0.5 * hour_sd * hour_sd + 0.5 * pooled_sd * pooled_sd)
        hour_sd = max(0.22, min(0.85, hour_sd))
        scale = hour_sd / pooled_sd
        median_log[h] = smoothed[h] + median_shift
        offsets[h] = {
            tau: (resid_q[tau] - median_shift) * scale for tau in LEVELS
        }
    return median_log, offsets


def persistence_forecast(p):
    """Fallback: flat median, spread widening like the house baseline."""
    z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817, 0.5: 0.0,
         0.75: 0.6744897501960817, 0.95: 1.6448536269514722}
    rows = []
    for h in range(2, 25):
        spread = 0.15 * math.sqrt(h - 1)
        rows.append({
            "hour": h,
            "q": [round(p * math.exp(z[tau] * spread), 2) for tau in LEVELS],
        })
    return {"hours": rows}


def forecast_case(case):
    known = case.get("known") or []
    p = 1.0
    if known:
        viewers = known[0].get("viewers")
        if viewers and viewers > 0:
            p = float(viewers)
    curves = comparable_curves(case)
    if len(curves) < 5:
        # Too little history to trust a fitted shape: stay flat.
        return persistence_forecast(p)
    median_log, offsets = fit_shape(curves, math.log(p))
    rows = []
    for h in range(2, 25):
        med = median_log[h]
        # Keep the median ratio in a sane band: never below a fifth of
        # hour 1, never above eight times it.
        med = max(math.log(0.2), min(math.log(8.0), med))
        qs = []
        prev = 0.0
        for tau in LEVELS:
            value = p * math.exp(med + offsets[h][tau])
            value = max(1.0, min(1e9, value))
            value = max(value, prev)
            qs.append(round(value, 2))
            prev = qs[-1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(in_dir) if name.endswith(".in")
    )
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            result = forecast_case(case)
        except Exception:
            result = persistence_forecast(1.0)
            known = case.get("known") or []
            if known and known[0].get("viewers"):
                result = persistence_forecast(float(known[0]["viewers"]))
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
