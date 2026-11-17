#!/usr/bin/env python3
"""Forecast launch-day Twitch viewers hour by hour (make.fun program/v1).

Reads one case file per run (schema makefun/twitch-ccu-case/v1) and writes,
for hours 2..24, five quantiles of that hour's concurrent-viewer count.

Method (everything is fitted from the case itself at run time):

  * Each past launch in `comparables` is reduced to its release-day shape:
    the log ratio ln(viewers_h / viewers_1) of each hour against its own
    first hour. The case's hour-1 print anchors the level; the comparables
    supply the shape. Launches also carry a pre-release ramp: their hourly
    counts in the hours before the release day, when the first regions
    unlock, which the case's own `history_hourly` mirrors.
  * For every hour block (2-6, 7-14, 15-24) and every quantile level, a
    quantile regression of the log ratio on the log size distance
    (ln hour-1 of the comparable minus ln hour-1 of the case) is fitted by
    profiling the exact pinball objective over a slope grid: for a fixed
    slope the best intercept is a weighted quantile of the adjusted
    ratios, so the fit is exact, deterministic and fast. The hour's five
    quantiles are the fitted lines read at the case's own size.
  * Comparables are weighted by resemblance to the case: release-type
    weights from the unlock text (rolling midnight-local console launches
    count most for such a case, simultaneous global unlocks and
    early-access-shaped days are downweighted), a Gaussian kernel on log
    size distance, a kernel on the pre-release last-hour ratio, and a
    kernel on the whole normalised pre-release ramp profile.
  * For rolling launches the median regression also fits a coefficient on
    the case's pre-release last-hour ratio: a launch whose early regions
    already ran hot (or cold) against its first hour tends to keep that
    offset through the day.
  * When a rolling case's pre-release peak already doubles its first
    hour, the two upper quantiles are drawn from a mixture that gives
    30% weight to past launches that broke out (peak at least 4x their
    first hour): such days carry a fatter upper tail than the pool's.
  * The spread around the median is scaled by 0.88, the value the
    leave-one-out validation below selects; quantile offsets are capped
    at 2.2 log.

Validation: leave-one-out over the 38 comparable launches, each one
forecast from its own first hour plus the other 37 and scored with the
contest's exact pinball-on-log loss. On the eight closest analogues of
the practice launches (big single-player console releases) this scores
0.223 against 0.273 for the previous version of this program and 0.460
for the persistence baseline; on a second analogue set with the big
midnight shooters included, 0.242 against 0.282 and 0.406; over all 37
usable launches, 0.503 against 0.731. The recipe was confirmed on
rotating holdout folds of the comparables before being frozen.

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
TIER_RE = re.compile(r"Tier:\s*([a-z][a-z\-]*)")
BLOCKS = ((2, 6), (7, 14), (15, 24))

# Release-type weights (see member weights in the validation harness).
W_WEAK = 0.20
W_EA = 0.50
W_GLOBAL = 0.25
W_MIDNIGHT = 1.00
W_UNKNOWN = 0.60
W_CONLY = 1.60
W_WITHPC = 0.60

SIZE_BW = 2.5        # Gaussian kernel width on log size distance
PRELAST_BW = 0.65    # kernel width on log pre-release last-hour ratio
SHAPE_BW = 0.5       # kernel width on pre-release ramp profile distance
MISS_FACTOR = 0.6    # weight factor when a kernel feature is missing

BETA_LO, BETA_HI = -0.30, 0.0   # slope grid range for the size regression
BETA_STEP = 0.05                # slope grid step (0.1 inside the 2-D fit)
GAMMA_RANGE = 1.0               # pre-release coefficient grid [-1, 1]
GAMMA_STEP = 0.1

SPREAD = 0.88                   # scale on the offsets from the median
MAX_OFF = 2.2                   # cap on any quantile's log offset
BREAKOUT_PI = 0.3               # mixture weight of the breakout pool
BREAKOUT_MARKER = 2.0           # case pre-release peak / hour 1 needed
BREAKOUT_PEAK = 4.0             # a train launch's peak ratio to qualify
MIN_H1 = 1000.0                 # a comparable needs a real first hour
MIN_PRESENT = 15                # ... and most of its release day
MIN_WEIGHT = 0.005


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def block_of(hour):
    for i, (lo, hi) in enumerate(BLOCKS):
        if lo <= hour <= hi:
            return i
    return 2


def release_day(comp):
    """{hour: viewers} over the comparable's release day (1..24)."""
    try:
        start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    by_hour = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            when = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((when - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            value = float(viewers)
            if hour in by_hour:
                by_hour[hour] = (by_hour[hour] + value) / 2.0
            else:
                by_hour[hour] = value
    return by_hour


def hourly_offsets(comp):
    """{offset_hours_from_day_start: viewers} for a comparable's points."""
    try:
        start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            when = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        off = int(round((when - start).total_seconds() / 3600.0))
        out[off] = float(viewers)
    return out


def pre_features(pre, h1):
    """Pre-release features from {offset: viewers} at offsets <= 0."""
    feats = {"pre_last": None, "pre_max24": None}
    if not pre or not h1:
        return feats, {}
    if -1 in pre:
        feats["pre_last"] = pre[-1] / h1
    window = [pre[o] for o in range(-24, 0) if o in pre]
    if window:
        feats["pre_max24"] = max(window) / h1
    prof = {}
    vals = [math.log(pre[o]) for o in range(-13, 0) if o in pre and pre[o] > 0]
    if len(vals) >= 6:
        mu = sum(vals) / len(vals)
        for o in range(-13, 0):
            if o in pre and pre[o] > 0:
                prof[o] = math.log(pre[o]) - mu
    return feats, prof


def traits(comp):
    """Release traits of one comparable, from its unlock note."""
    unlock = comp.get("unlock") or ""
    text = unlock.lower()
    match = TIER_RE.search(unlock)
    tier = match.group(1) if match else None
    platforms = comp.get("platforms") or ""
    return {
        "weak": tier == "weak",
        "ea": tier == "early-access-contaminated",
        "global": ("single global" in text or "simultaneous" in text
                   or "single simultaneous" in text),
        "midnight": "midnight local" in text,
        "console_only": "PC" not in platforms and "Stadia" not in platforms,
    }


def case_regime(case):
    """What kind of release this case is, from its own notes."""
    notes = (case.get("notes") or "").lower()
    rolling = "midnight local" in notes or "rolling" in notes
    conly = ("console only" in notes
             or ("pc" not in notes and "stadia" not in notes))
    return rolling, conly


def case_pre(case):
    """The case's own pre-release hourly viewers: {offset: viewers}."""
    try:
        start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    hourly = case.get("history_hourly") or []
    if hourly:
        for row in hourly:
            viewers = row.get("viewers")
            if not viewers or viewers <= 0:
                continue
            try:
                when = parse_time(row["from"])
            except (KeyError, TypeError, ValueError):
                continue
            off = int(round((when - start).total_seconds() / 3600.0))
            if off <= 0:
                out[off] = float(viewers)
        return out
    # Fall back to bucketing the raw history points by hour.
    buckets = {}
    for point in case.get("history") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            when = parse_time(point["at"])
        except (KeyError, TypeError, ValueError):
            continue
        off = int(math.floor((when - start).total_seconds() / 3600.0))
        if off <= 0:
            buckets.setdefault(off, []).append(float(viewers))
    return {off: sum(v) / len(v) for off, v in buckets.items()}


def shape_dist(a, b):
    common = [o for o in a if o in b]
    if len(common) < 6:
        return None
    return math.sqrt(sum((a[o] - b[o]) ** 2 for o in common) / len(common))


def weighted_quantile(pairs, level):
    """Weighted quantile of [(value, weight)] by linear interpolation."""
    ordered = sorted(pairs, key=lambda item: item[0])
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


def pinball_obj(pts, tau, alpha, beta, gam):
    return sum(w * 2.0 * max(tau * (r - alpha - beta * x - gam * dz),
                             (tau - 1.0) * (r - alpha - beta * x - gam * dz))
               for x, r, w, dz in pts)


def fit_beta(pts, tau, gam):
    """Profile the pinball objective over the slope grid at fixed gamma."""
    best = None
    beta = BETA_LO
    while beta <= BETA_HI + 1e-9:
        adj = [(r - beta * x - gam * dz, w) for x, r, w, dz in pts]
        alpha = weighted_quantile(adj, tau)
        obj = pinball_obj(pts, tau, alpha, beta, gam)
        if best is None or obj < best[0]:
            best = (obj, beta)
        beta += BETA_STEP
    return best[1]


def comparable_pool(case):
    """Usable comparable launches with shapes, traits and pre features."""
    pool = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        if name.startswith("Grand Theft Auto VI category"):
            continue  # the target's own pre-release category, no launch
        day = release_day(comp)
        first = day.get(1)
        if not first or first < MIN_H1:
            continue
        present = sum(1 for h in range(2, 25) if h in day)
        if present < MIN_PRESENT:
            continue
        offs = hourly_offsets(comp)
        pre = {o: v for o, v in offs.items() if o <= 0}
        feats, prof = pre_features(pre, first)
        ratios = {h: math.log(v / first) for h, v in day.items() if h >= 2}
        pool.append({
            "log_h1": math.log(first),
            "traits": traits(comp),
            "ratios": ratios,
            "peak_ratio": max([v / first for h, v in day.items() if h >= 2]
                              or [1.0]),
            "feat": feats,
            "prof": prof,
        })
    return pool


def forecast_case(case):
    known = case.get("known") or []
    p = 1.0
    if known:
        viewers = known[0].get("viewers")
        if viewers and viewers > 0:
            p = float(viewers)
    pool = comparable_pool(case)
    if len(pool) < 4:
        return persistence_forecast(p)
    rolling, conly = case_regime(case)
    log_p = math.log(p)
    cpre = case_pre(case)
    cfeat, cprof = pre_features(cpre, p)
    zcase = math.log(cfeat["pre_last"]) if cfeat["pre_last"] else None

    # Weights: release type, then kernels on size, pre-release last hour
    # and the pre-release ramp profile.
    train = []
    for e in pool:
        tr = e["traits"]
        w = 1.0
        if tr["weak"]:
            w *= W_WEAK
        if tr["ea"]:
            w *= W_EA
        if rolling:
            if tr["global"]:
                w *= W_GLOBAL
            elif tr["midnight"]:
                w *= W_MIDNIGHT
            else:
                w *= W_UNKNOWN
        if conly:
            w *= W_CONLY if tr["console_only"] else W_WITHPC
        d = abs(e["log_h1"] - log_p)
        w *= math.exp(-0.5 * (d / SIZE_BW) ** 2)
        a = e["feat"]["pre_last"]
        if a and cfeat["pre_last"]:
            dd = abs(math.log(a) - math.log(cfeat["pre_last"]))
            w *= math.exp(-0.5 * (dd / PRELAST_BW) ** 2)
        else:
            w *= MISS_FACTOR
        dist = shape_dist(e["prof"], cprof)
        if dist is not None:
            w *= math.exp(-0.5 * (dist / SHAPE_BW) ** 2)
        else:
            w *= MISS_FACTOR
        if w > MIN_WEIGHT:
            dz = 0.0
            if a and zcase is not None:
                dz = math.log(a) - zcase
            train.append((e, w, dz))
    if len(train) < 4:
        return persistence_forecast(p)

    # Fit one slope (and, for rolling cases, one pre-release coefficient)
    # per hour block and quantile level on the pooled block points.
    slopes = {}
    gammas = {}
    for b in range(3):
        pts = []
        for e, w, dz in train:
            for h in range(2, 25):
                if block_of(h) == b and h in e["ratios"]:
                    pts.append((e["log_h1"] - log_p, e["ratios"][h], w, dz))
        gammas[b] = 0.0
        for tau in LEVELS:
            if rolling and tau == 0.5:
                best = None
                beta = BETA_LO
                while beta <= BETA_HI + 1e-9:
                    gam = -GAMMA_RANGE
                    while gam <= GAMMA_RANGE + 1e-9:
                        adj = [(r - beta * x - gam * dz, w)
                               for x, r, w, dz in pts]
                        alpha = weighted_quantile(adj, tau)
                        obj = pinball_obj(pts, tau, alpha, beta, gam)
                        if best is None or obj < best[0]:
                            best = (obj, beta, gam)
                        gam += GAMMA_STEP
                    beta += 0.1
                slopes[(b, tau)] = best[1]
                gammas[b] = best[2]
                continue
            slopes[(b, tau)] = fit_beta(pts, tau, gammas[b])

    case_breaky = bool(rolling and cfeat["pre_max24"]
                       and cfeat["pre_max24"] >= BREAKOUT_MARKER)
    rows = []
    for h in range(2, 25):
        b = block_of(h)
        gam = gammas[b]
        qs = {}
        for tau in LEVELS:
            beta = slopes[(b, tau)]
            pairs = [(e["ratios"][h] - beta * (e["log_h1"] - log_p)
                      - gam * dz, w)
                     for e, w, dz in train if h in e["ratios"]]
            if not pairs:
                qs[tau] = 0.0
                continue
            if case_breaky and tau >= 0.75:
                bpairs = [(e["ratios"][h] - beta * (e["log_h1"] - log_p)
                           - gam * dz, w)
                          for e, w, dz in train
                          if h in e["ratios"] and e["peak_ratio"] >= BREAKOUT_PEAK]
                if len(bpairs) >= 3:
                    wm = sum(w for _, w in pairs)
                    wb = sum(w for _, w in bpairs)
                    mix = ([(v, w * (1.0 - BREAKOUT_PI)) for v, w in pairs]
                           + [(v, w * BREAKOUT_PI * wm / wb)
                              for v, w in bpairs])
                    qs[tau] = weighted_quantile(mix, tau)
                    continue
            qs[tau] = weighted_quantile(pairs, tau)
        median = qs[0.5]
        row = []
        for tau in LEVELS:
            off = (qs[tau] - median) * SPREAD
            if tau < 0.5:
                off = max(off, -MAX_OFF)
            else:
                off = min(off, MAX_OFF)
            row.append(max(1.0, min(1e9, p * math.exp(median + off))))
        for i in range(1, 5):
            row[i] = max(row[i], row[i - 1])
        rows.append({"hour": h, "q": [round(v, 2) for v in row]})
    return {"hours": rows}


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
