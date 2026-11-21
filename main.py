"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, five quantiles of its
earn rate on Exophase at the case's target, from the house's reads in the case file.

Method (standard library only, deterministic, no network):
  * The case history is collapsed to distinct states: consecutive reads with the
    same players-tracked count and the same rates are one state. Exophase only
    changes a game's counts when it refreshes, hours apart, so between refreshes
    the page is frozen and the extra reads carry no signal.
  * Per trophy, on the logit scale, a recency-weighted least-squares line of
    logit(rate) against ln(1 + hours since release) is fitted over the distinct
    states (half-life 24 h). A trophy's earn rate cannot grow past the share
    of players who ever do what it asks, so for a rising trophy the fit is
    also tried with a ceiling: logit(rate / L) against the same x, with L
    grid-searched, and the ceiling is kept only when it fits clearly better
    than no ceiling. Slopes are shrunk toward the cross-trophy median slope
    in the same space, so a trophy with a noisy history borrows strength from
    the rest. The median forecast reads the fitted line at the target; the
    spread is the line's prediction interval combined with a random-walk term
    that grows with the horizon (0.5 logit per sqrt day, the house baseline's
    rate).
  * When a trophy's history shows no movement at all (a single distinct
    state), the line cannot be fitted. Then the forecast is a two-component
    mixture: with probability p the page is refreshed before the target, and
    a refresh dilutes every rate (new players hold far fewer trophies than
    the tracked cohort). The launch-night refresh described in the rules
    took players tracked 638 -> 3,868 and cut every rate but the one at 100%
    by about 1.5 logit; that is the prior for a refresh's size. p grows
    with the time since the page last changed, measured at the target.
  * A trophy at 100% in every state is treated on its own: its median stays
    at the top, with a left tail that widens with the horizon, because a
    100% rate can only fall as the tracked population grows.

Local scores, measured with this file (the practice answers are reconstructed
from the aggregates the rules publish for the 04:30 refresh -- players tracked
3,868, trophies earned 4,276, trophy 2 staying at 100% -- not the sealed
answers, so these are indicative, not exact):
  practice p00 loss 0.353, p01 loss 0.270 (house baseline 1.404 on the same
  reconstruction; the baseline's stated figures are loss ~1.51, miss ~0.68pp,
  coverage 0, and the reconstruction reproduces its 0.68pp miss exactly).
  synthetic final-like case, saturating rates: loss 0.255 vs baseline 0.316.
  synthetic final-like case, non-saturating logit-linear rates: loss 0.229
  vs baseline 0.458. The synthetic cases are my own simulations, built only
  to exercise the trend path the flat practice histories cannot test.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Random-walk spread of a rate's move, logit units per sqrt day (house baseline).
SIGMA_RW = 0.5
# Recency half-life of a read in the trend fit, hours.
HALF_LIFE_H = 24.0
# Prior size of a refresh's dilution, logit units (launch-night refresh, rules).
REFRESH_SHIFT = 1.5
# A refresh's gap is a few hours; p grows with the time since the page last
# changed, measured at the target.
REFRESH_P0_H = 2.5
REFRESH_P_SCALE_H = 4.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as the house baseline does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(p: float, comp):
    """Quantile p of a mixture of normals comp = [(weight, mean, sd)]."""
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = sum(w * norm_cdf((mid - m) / s) for w, m, s in comp)
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def distinct_states(history):
    """Collapse consecutive reads with identical counts and rates to one state."""
    states = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if states and states[-1][0] == key:
            continue
        states.append((key, read))
    return [read for _, read in states]


def fit_line(xs, ys, ws):
    """Recency-weighted least squares y ~ a + b*x. Returns (a, b, xbar, sxx, n_eff)."""
    sw = sum(ws)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return ybar, 0.0, xbar, sxx, (sw * sw) / sum(w * w for w in ws)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    n_eff = (sw * sw) / sum(w * w for w in ws)
    return a, b, xbar, sxx, n_eff


def weighted_sse(xs, ys, ws, a, b):
    return sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))


def fit_ceiling(xs, rates_pct, ws):
    """Grid-search a ceiling L for logit(rate / L) ~ a + b*x.

    Returns (L, a, b, sse, sse_plain): L is the best ceiling in percent, or
    100.0 when no ceiling beats the plain logit fit (L = 100) clearly.
    Only called for a rising series; a ceiling below the largest observed
    rate is impossible, so the grid starts just above it.
    """
    rmax = max(rates_pct)
    # Plain fit (L = 100): y = logit(rate) with rate in percent.
    ys_plain = [logit(r) for r in rates_pct]
    a0, b0, _, _, _ = fit_line(xs, ys_plain, ws)
    sse_plain = weighted_sse(xs, ys_plain, ws, a0, b0)
    best = (100.0, a0, b0, sse_plain)
    lo = math.log(max(rmax * 1.02, 0.02))
    hi = math.log(99.99)
    if hi - lo < 1e-9:
        return best + (sse_plain,)
    n_grid = 24
    for k in range(n_grid):
        L = math.exp(lo + (hi - lo) * k / (n_grid - 1))
        ys = [math.log(min(0.999, max(0.001, r / L)) / (1 - min(0.999, max(0.001, r / L))))
              for r in rates_pct]
        a, b, _, _, _ = fit_line(xs, ys, ws)
        sse = weighted_sse(xs, ys, ws, a, b)
        if sse < best[3]:
            best = (L, a, b, sse)
    # Keep the ceiling only when it fits clearly better: one extra parameter
    # must earn its keep, or the extrapolation would chase noise.
    if best[0] < 100.0 and best[3] < 0.85 * sse_plain:
        return best + (sse_plain,)
    return (100.0, a0, b0, sse_plain, sse_plain)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        # No reads at all: a flat, very wide forecast is the honest answer.
        return {"trophies": [{"id": t["id"], "q": [0.05, 0.5, 1.0, 2.0, 10.0]}
                             for t in case["trophies"]]}
    states = distinct_states(history)
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log(1.0 + (target - release).total_seconds() / 3600.0)
    last = states[-1]
    t_last = parse_time(last["at"])

    # Per-trophy series over the distinct states: (x, rate percent, weight).
    series = []
    for i, trophy in enumerate(case["trophies"]):
        pts = []
        for st in states:
            rate = st["rates"][i] if i < len(st["rates"]) else None
            if rate is None:
                continue
            t = parse_time(st["at"])
            x = math.log(1.0 + (t - release).total_seconds() / 3600.0)
            w = 0.5 ** ((t_last - t).total_seconds() / 3600.0 / HALF_LIFE_H)
            pts.append((x, rate, w))
        series.append(pts)

    # First pass: plain and ceiling fits per trophy, and the median slope in
    # each space, for shrinkage.
    plain_slopes, ceil_slopes = [], []
    trend = []  # per trophy: None or dict with the fitted model
    for pts in series:
        if len(pts) >= 2 and max(p[0] for p in pts) - min(p[0] for p in pts) > 0.05:
            xs = [p[0] for p in pts]
            rates = [p[1] for p in pts]
            ws = [p[2] for p in pts]
            a0, b0, xbar, sxx, n_eff = fit_line(xs, [logit(r) for r in rates], ws)
            plain_slopes.append(b0)
            if b0 > 0 and len(pts) >= 3:
                L, a_c, b_c, sse_c, sse_plain = fit_ceiling(xs, rates, ws)
                if L < 100.0:
                    ceil_slopes.append(b_c)
                    trend.append({"kind": "ceil", "L": L, "a": a_c, "b": b_c,
                                  "xbar": xbar, "sxx": sxx, "n_eff": n_eff,
                                  "xs": xs, "rates": rates, "ws": ws,
                                  "sse": sse_c})
                    continue
            trend.append({"kind": "plain", "a": a0, "b": b0,
                          "xbar": xbar, "sxx": sxx, "n_eff": n_eff,
                          "xs": xs, "rates": rates, "ws": ws})
        else:
            trend.append(None)
    plain_slopes.sort()
    b_med_plain = plain_slopes[len(plain_slopes) // 2] if plain_slopes else 0.0
    ceil_slopes.sort()
    b_med_ceil = ceil_slopes[len(ceil_slopes) // 2] if ceil_slopes else 0.0

    # Time since the page last changed, measured at the target (for the
    # no-signal fallback's refresh probability).
    first_t = parse_time(states[0]["at"])
    hours_since_change_at_target = (target - first_t).total_seconds() / 3600.0
    p_refresh = min(0.9, max(0.05, (hours_since_change_at_target - REFRESH_P0_H) / REFRESH_P_SCALE_H))
    sd_rw = SIGMA_RW * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = None
        for st in reversed(states):
            if i < len(st["rates"]) and st["rates"][i] is not None:
                last_rate = st["rates"][i]
                break
        if last_rate is None:
            last_rate = 0.0
        all_100 = all(
            i < len(st["rates"]) and st["rates"][i] == 100 for st in states
        )
        if all_100:
            # A trophy every tracked player holds: median stays at the top,
            # with a left tail that widens with the horizon.
            sd = 1.0 + 1.4 * math.sqrt(horizon_days)
            lg = logit(99.99)
            qs = [inv_logit(lg + z * sd) for z in Z]
        elif trend[i] is not None:
            tr = trend[i]
            xs, rates, ws = tr["xs"], tr["rates"], tr["ws"]
            xbar, sxx, n_eff = tr["xbar"], tr["sxx"], tr["n_eff"]
            w_shrink = n_eff / (n_eff + 3.0)
            if tr["kind"] == "ceil":
                b_s = w_shrink * tr["b"] + (1.0 - w_shrink) * b_med_ceil
                ys = [math.log(min(0.999, max(0.001, r / tr["L"]))
                             / (1 - min(0.999, max(0.001, r / tr["L"])))) for r in rates]
            else:
                b_s = w_shrink * tr["b"] + (1.0 - w_shrink) * b_med_plain
                ys = [logit(r) for r in rates]
            ybar_w = sum(w * y for y, w in zip(ys, ws)) / sum(ws)
            a_s = ybar_w - b_s * xbar
            y_med = a_s + b_s * x_target
            resid = [y - (tr["a"] + tr["b"] * x) for x, y in zip(xs, ys)]
            sw = sum(ws)
            resid_var = sum(w * r * r for r, w in zip(resid, ws)) / max(sw - 2.0, 1.0)
            resid_var = min(resid_var, 4.0)
            pred_var = resid_var * (1.0 + 1.0 / max(n_eff, 1.0)
                                    + (x_target - xbar) ** 2 / sxx) if sxx > 1e-12 else resid_var
            sd = math.sqrt(pred_var + sd_rw ** 2 + 0.15 ** 2)
            sd = min(max(sd, 0.4), 3.5)
            if tr["kind"] == "ceil":
                qs = [tr["L"] * inv_logit(y_med + z * sd) / 100.0 for z in Z]
            else:
                qs = [inv_logit(y_med + z * sd) for z in Z]
        else:
            # No movement in the history: persistence, mixed with the chance
            # of a refresh diluting the rates before the target.
            lg = logit(last_rate)
            sd0 = math.sqrt(sd_rw ** 2 + 0.15 ** 2)
            comp = [(1.0 - p_refresh, lg, max(sd0, 0.25)),
                    (p_refresh, lg - REFRESH_SHIFT, 0.5)]
            qs = [inv_logit(mixture_quantile(tau, comp)) for tau in LEVELS]
        qs = [round4(min(100.0, max(0.01, q))) for q in qs]
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
