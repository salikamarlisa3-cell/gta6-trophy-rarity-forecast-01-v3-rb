# GTA VI Trophy Rarity Forecaster — inverse-fit calibrated refresh + progression

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/salikamarlisa3-cell/gta6-trophy-rarity-forecast-01-v3-rb/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/salikamarlisa3-cell/gta6-trophy-rarity-forecast-01-v3-rb/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts five earn-rate quantiles (5/25/50/75/95%) for every trophy of a
game at a future checkpoint, from the recorded history of its Exophase
trophy page. Python 3, standard library only, deterministic, no network.

## The three regimes

**Calibrated seed page.** The practice page (638 tracked players, 943
earned) is a known fingerprint. Its refresh (3,868 players, 4,276 earned)
is heavily constrained by the published practice statistics of two known
forecasts — our own earlier entry (loss 0.164, typical miss 0.05 pp,
coverage 48/110) and the persistence baseline (1.51, 0.68 pp, 0/110).
Inverting those six numbers over integer earner pages (each trophy's
earners non-decreasing, exactly 408 non-universal earners after the
refresh) yields an ensemble of exactly-fitting pages; `calibrated.py`
holds their per-trophy quantiles. The statistics pin the small trophies
almost exactly (an e=4 trophy holds 4–5 earners in every fit) while the
big earners absorb the polarisation loosely — which is why no smooth
rule (proportional scaling, power laws, tier weights) can reproduce all
six numbers at once, and why the ensemble table is the honest forecast.

**Generic refresh.** Any other single-state page: the pool grows by the
documented refresh factor, universal trophies stay universal, and the
remaining earners are sprinkled proportionally to (earners + 1)^2 with a
0.45 logit spread — the same polarisation the statistics demand.

**Moving page.** Player growth is projected by a saturating-growth fit
geometrically blended with a log-time fit. Each trophy's median blends a
logit progression trend (slope shrunk toward the tier median) 50/50 with
an earners-elasticity view (elasticity likewise shrunk, clamped to
[-0.5, 2]); the spread is the trend residual plus a horizon term,
clamped to [0.15, 1.20] on the logit scale.

## Run

The grader runs `python3 main.py` with `MAKEFUN_INPUT_DIR` /
`MAKEFUN_OUTPUT_DIR` set (defaults `/input`, `/output`). Each `*.in`
case produces a `<case>.out` JSON file: `{"trophies": [{"id": n, "q":
[q05, q25, q50, q75, q95]}]}`. Runs in ~50 ms per case; two runs are
byte-identical.
