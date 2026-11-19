# GTA VI Trophy Rarity Forecast — earners / players decomposition

A forecaster for make.fun's `gta6-trophy-rarity-forecast-01` bounty: given
Exophase's trophy page reads for Grand Theft Auto VI (PS5) up to a
checkpoint, it forecasts every trophy's earn rate at the target
(2026-11-26T00:00:00Z) as five quantiles, scored by pinball loss on the
logit scale.

## The idea: a rate is a ratio, so forecast its two sides

A trophy's earn rate is `100 * earners / players`. On launch week both
sides move fast, for different reasons:

- **Players tracked** grows as new players sync their profiles — a launch
  surge that dilutes every rate at once.
- **Earners** grow as those players play through the story, lifting early
  trophies first and late trophies later.

Fitting a curve to the rate itself confuses the two. A rate that is
falling only because new players keep arriving looks like a trophy nobody
earns; a rate that is flat because the page has not refreshed looks like
nothing is happening. (That is exactly what launch night looked like:
Exophase showed 638 players and identical rates for hours, then refreshed
to 3,868 players with every rate but one collapsed — the newly tracked
players held far fewer trophies.)

So `main.ts` fits each side separately, in log-time
`x = ln(1 + hours since 2026-11-19T00:00Z)`:

1. Reconstruct per-trophy earners `E = rate/100 * players` for each read.
2. Fit a least-squares line of `ln(players)` vs `x`, and per trophy a
   line of `ln(E)` vs `x`. Read both at the target.
3. Earners are cumulative, so their forecast never falls below the last
   read; players are capped (at most ~55x growth beyond the last read)
   and never fall either; earners are capped at players.
4. Forecast rate `= 100 * E_target / P_target`.
5. Spread combines a base random-walk sigma with each trophy's fit
   residual. A trophy already at ~100% (everyone earns it at once) gets a
   tight spread — it stays at 100%.

Two details of the data shape the fitting:

- **Frozen pages repeat verbatim.** While Exophase has not refreshed, the
  page repeats, so consecutive identical reads are collapsed to one
  observation before fitting; feeding them all to a line fit would drag
  every slope toward zero.
- **The first state's timestamp does not date it.** The page as found
  when recording started may have shown those values for hours already,
  and as the farthest point in log-time it would lever every slope down.
  It is dropped from the fits (the last read still anchors levels).

**Flat history** (every read identical, or fewer than 3 reads) carries no
trend, so the program falls back to persistence — median = last rate —
with a wider spread (see below). It never invents a trend from flat data.

## How to run

Node 22, standard library only, no network, deterministic (two runs write
identical bytes, ~0.15 s per case, well under the 8 s cap):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory (`$MAKEFUN_INPUT_DIR`, else
`/input`) and writes `<id>.out` to the output directory
(`$MAKEFUN_OUTPUT_DIR`, else `/output`): `{"trophies":[{"id":n,
"q":[q05,q25,q50,q75,q95]}, ...]}`, one row per trophy, quantiles
non-decreasing, 0–100.

## Validation (mean pinball loss on the logit scale, lower is better)

| Case | This forecaster | Persistence baseline |
| ---- | --------------: | -------------------: |
| v00 — real 8 h dilution test (checkpoint 2026-11-19T03:30Z, live Exophase read 11:45Z: players 638 → 3,868, rates collapsed ~5×) | **0.881** | 1.389 |
| p00 / p01 — practice cases vs the refreshed-page rates | **1.243** | 1.508 |
| Synthetic final-like case, clean log-time growth (players 3,868 → 51,270; per-trophy earner slopes 0.4–1.8) | **0.218** | 0.304 |
| Same synthetic case with 15% multiplicative noise on the reads | **0.251** | 0.304 |

Practice outputs are valid (55 rows, five non-decreasing quantiles each)
and byte-identical across two runs.

### How the constants were chosen

- **`SIGMA0 = 0.5`** (decomposition path) and **`FLAT_SIGMA0 = 2.0`**
  (flat fallback) are separate on purpose. On the synthetic final-like
  cases the model path is best at 0.5 (0.218 vs 0.396 at 1.0), and a
  spread-sensitivity analysis puts 0.5 near-optimal for the final's
  4.5-day horizon across plausible median-error scales — a bigger sigma
  only chases v00's short-horizon refresh. But flat history is genuine
  ignorance: the next refresh moved rates ~−1.6 logit in 8 hours on v00,
  so the fallback needs the wider 2.0, which is what beats persistence
  there (0.881 vs 1.389).
- **`DAMP = 1.0`** (no extra damping): the log-time clock already
  saturates growth — `x` rises only ~25% from the deadline to the
  target — and the synthetic cases score better undamped (0.218) than at
  0.75 (0.249).
- **`SHRINK_K = 0`** (no shrinkage toward the median slope): true slopes
  genuinely differ per trophy, the per-trophy fits are good enough, and
  shrinkage only adds bias — clean 0.218 / 0.334 and noisy 0.251 / 0.322
  at K = 0 / 3, getting monotonically worse as K grows.

One honest note: on v00 the medians equal the last rates (the history is
perfectly flat, so the spec's flat fallback is persistence by
construction) — the v00 win comes from the widened spread, not from a
shifted median. On the final case, where the history carries real
refreshes, the decomposition moves the medians (in the synthetic
final-like case, early trophies rise and late ones fall, as the
earners/players split implies).

## License

MIT — see `LICENSE` (the kit starter's license, copied verbatim).
