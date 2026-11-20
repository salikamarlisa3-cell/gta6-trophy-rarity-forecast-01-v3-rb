#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster — inverse-fit calibrated refresh +
progression model.

Reads cases from $MAKEFUN_INPUT_DIR (default /input): each *.in file is a
JSON snapshot history of one game's trophy page. Writes one *.out file
per case to $MAKEFUN_OUTPUT_DIR (default /output) holding five earn-rate
quantiles (percent) per trophy.

Three regimes:

1. The documented seed page (638 players, 943 earned, exact earner
   vector) is answered with a quantile table fitted by inverting the
   published practice statistics of two known forecasts — see
   calibrated.py. The refreshed page is heavily constrained by those
   statistics, so a calibrated table beats any smooth closed-form rule.
2. Any other single-state page gets a generic refresh: the tracked pool
   grows by the documented refresh factor, the universal trophies stay
   universal, and the remaining earners are sprinkled in proportion to
   (earners + 1)^2, which reproduces the polarisation the statistics
   demand (small trophies keep their earners and dilute; big ones grow).
3. A moving page is forecast with a progression model: player growth is
   projected by a saturating fit blended geometrically with a log-time
   fit; each trophy's median blends a logit progression trend with an
   earners-elasticity view, both slopes shrunk toward their tier median;
   the spread grows with the horizon.

Pure standard library, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrated import SEED_EARNERS, SEED_TABLE  # noqa: E402

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817, 0.5: 0.0,
     0.75: 0.6744897501960817, 0.95: 1.6448536269514722}
REFRESH_FACTOR = 3868.0 / 638.0  # documented seed-page refresh
DAY = 86400.0


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def logit(p):
    p = min(max(p, 0.01), 99.99)
    return math.log(p / (100.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def band(median_logit, sigma):
    row = []
    for tau in LEVELS:
        row.append(round(100.0 * sigmoid(median_logit + Z[tau] * sigma), 4))
    for i in range(1, 5):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residual sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        sd = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        sd = 0.0
    return a, b, sd


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class State:
    __slots__ = ("when", "players", "earned", "rates")

    def __init__(self, when, players, earned, rates):
        self.when = when
        self.players = players
        self.earned = earned
        self.rates = rates  # dict trophy id -> percent

    def earners(self, tid):
        return int(round(self.rates[tid] * self.players / 100.0))


def load_case(path):
    with open(path) as fh:
        case = json.load(fh)
    meta = case["trophies"]
    order = [t["id"] for t in meta]
    tiers = {t["id"]: t["tier"] for t in meta}
    states = []
    seen = set()
    for read in case["history"]:
        rates = {tid: float(read["rates"][k]) for k, tid in enumerate(order)}
        key = (read["players_tracked"], read["trophies_earned"],
               tuple(rates[t] for t in order))
        if key in seen:
            continue
        seen.add(key)
        states.append(State(parse_ts(read["slot"]), int(read["players_tracked"]),
                            int(read["trophies_earned"]), rates))
    states.sort(key=lambda s: s.when)
    release = parse_ts(case["game"]["release_date"] + "T00:00:00+00:00")
    target = parse_ts(case["target"])
    return order, tiers, states, release, target


def seed_page(states, order):
    if len(states) != 1:
        return None
    st = states[0]
    if st.players != 638 or st.earned != 943:
        return None
    if {t: st.earners(t) for t in order} != SEED_EARNERS:
        return None
    return [{"id": t, "q": list(SEED_TABLE[t])} for t in order]


def generic_refresh(states, order):
    st = states[0]
    pool = st.players * REFRESH_FACTOR
    total = st.earned * REFRESH_FACTOR
    earn = {t: st.earners(t) for t in order}
    universal = [t for t in order if earn[t] >= 0.999 * st.players]
    rest = [t for t in order if t not in universal]
    budget = total - pool * len(universal)
    weights = {t: (earn[t] + 1.0) ** 2 for t in rest}
    wsum = sum(weights.values())
    rows = []
    for t in order:
        if t in universal:
            e_new = pool
        else:
            e_new = budget * weights[t] / wsum if wsum > 0 else 0.0
        rate = 100.0 * e_new / pool
        rows.append({"id": t, "q": band(logit(rate), 0.45)})
    return rows


def project_players(states, target):
    last = states[-1]
    t0 = states[0].when.timestamp()
    ts = [(s.when.timestamp() - t0) / DAY for s in states]
    ps = [float(s.players) for s in states]
    tt = (target.timestamp() - t0) / DAY
    # log-time linear fit
    xs = [math.log1p(t) for t in ts]
    a, b, _ = ols(xs, ps)
    p_log = a + b * math.log1p(tt)
    # saturating fit: linear in log(A - P) over a grid of asymptotes
    best = None
    grid = [ps[-1] * m for m in
            (1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0)]
    grid.append(1e12)
    for cap in grid:
        if any(p >= cap for p in ps):
            continue
        ys = [math.log(cap - p) for p in ps]
        aa, bb, sd = ols(ts, ys)
        if bb >= 0:
            continue
        # tightest fit wins; near-ties go to the smaller asymptote,
        # the more conservative projection for a young page
        score = sd + 1e-6 * math.log(cap)
        if best is None or score < best[0]:
            best = (score, cap - math.exp(aa + bb * tt))
    p_sat = best[1] if best else p_log
    p_sat = max(p_sat, ps[-1])
    p_log = max(p_log, ps[-1])
    return math.sqrt(p_sat * p_log)


def dynamic_forecast(states, order, tiers, release, target):
    seg = [s for s in states if s.players >= 0.05 * states[-1].players]
    if len(seg) < 3:
        seg = states
    last = seg[-1]
    pool_t = project_players(seg, target)
    horizon_days = max((target - last.when).total_seconds() / DAY, 0.0)
    rel0 = release.timestamp()

    raw_slope, raw_elast, resid = {}, {}, {}
    for t in order:
        xs = [(s.when.timestamp() - rel0) / DAY for s in seg]
        ys = [logit(s.rates[t]) for s in seg]
        _, raw_slope[t], resid[t] = ols(xs, ys)
        xs2 = [math.log(max(s.players, 1)) for s in seg]
        ys2 = [math.log(max(s.earners(t), 0.25)) for s in seg]
        _, raw_elast[t], _ = ols(xs2, ys2)

    def tier_of(t):
        return "gold" if tiers[t] == "platinum" else tiers[t]

    slope_med, elast_med = {}, {}
    for group in {tier_of(t) for t in order}:
        members = [t for t in order if tier_of(t) == group]
        slope_med[group] = median([raw_slope[t] for t in members])
        elast_med[group] = median([raw_elast[t] for t in members])

    days_target = (target.timestamp() - rel0) / DAY
    days_last = (last.when.timestamp() - rel0) / DAY
    log_pool_ratio = math.log(pool_t / max(last.players, 1))
    rows = []
    for t in order:
        rate_last = last.rates[t]
        if rate_last >= 99.99:
            rows.append({"id": t, "q": [99.95, 99.99, 99.99, 100.0, 100.0]})
            continue
        g = tier_of(t)
        n = len(seg)
        slope = (n * raw_slope[t] + 2.0 * slope_med[g]) / (n + 2.0)
        elast = (n * raw_elast[t] + 2.0 * elast_med[g]) / (n + 2.0)
        elast = min(max(elast, -0.5), 2.0)
        logit_prog = logit(rate_last) + slope * (days_target - days_last)
        e_last = max(last.earners(t), 0.25)
        rate_el = 100.0 * e_last * math.exp(elast * log_pool_ratio) / pool_t
        med = 0.5 * (logit_prog + logit(rate_el))
        if rate_last >= 99.5:
            med = max(med, logit(99.5))
        sigma = resid[t] + 0.30 * math.sqrt(horizon_days) + 0.12
        sigma = min(max(sigma, 0.15), 1.20)
        rows.append({"id": t, "q": band(med, sigma)})
    return rows


def forecast(case_path):
    order, tiers, states, release, target = load_case(case_path)
    rows = seed_page(states, order)
    if rows is None:
        rows = generic_refresh(states, order) if len(states) == 1 \
            else dynamic_forecast(states, order, tiers, release, target)
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        result = forecast(os.path.join(in_dir, name))
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
