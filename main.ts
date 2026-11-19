// GTA VI trophy rarity forecast — earners / players decomposition forecaster.
//
// A trophy's earn rate is a ratio: rate = 100 * earners / players. On launch
// week both sides move fast, and for different reasons. Players tracked grows
// as new players sync their profiles — a launch surge that dilutes every
// rate at once. Earners grow as those players play through the story, which
// lifts early trophies first and late trophies later. Fitting a curve to the
// rate itself confuses the two movements: a rate that is falling only because
// new players keep arriving looks like a trophy nobody earns, and a rate that
// is flat because the page has not refreshed looks like nothing is happening.
//
// So this program fits each side separately. For every read it reconstructs
// per-trophy earners E = rate/100 * players_tracked, then fits a straight
// line of ln(E) against ln(1 + hours since release), and a straight line of
// ln(players) against the same clock. Both lines are read at the target, the
// earners' slope is shrunk toward the cross-trophy median slope (tiny counts
// are mostly rounding noise), the extrapolated growth is damped (progress
// saturates), and the forecast rate is 100 * E_target / P_target. The spread
// combines a base random-walk sigma with each trophy's fit residual.
//
// When the history is flat — every read identical, which is what launch night
// looks like while Exophase has not refreshed — no trend can be read from it,
// so the program falls back to persistence (median = last rate) with the
// base spread. It never invents a trend from flat data.
//
// Node 22, standard library only, no network, deterministic.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ---------------------------------------------------------------------------
// Tunables (chosen on the validation case v00; see README.md).
// ---------------------------------------------------------------------------

/** Base random-walk sigma, in logit units per square-root day, for the
 *  decomposition path (history with a readable trend).
 *  Chosen on the synthetic validation cases (see README.md): 0.5 beats
 *  persistence there (0.218 vs 0.304) and is near-optimal for the final's
 *  4.5-day horizon across plausible median-error scales. Larger values
 *  only chase v00's short-horizon refresh and blow the final's spread up. */
const SIGMA0 = 0.5;
/** Spread sigma for the flat-history fallback (no trend readable at all).
 *  A frozen page can jump a long way at the next refresh — v00's jump was
 *  about -1.6 logit in 8 hours — so the fallback needs a much wider spread
 *  than the model path. 2.0 scores 0.881 on v00 against persistence's 1.389. */
const FLAT_SIGMA0 = 2.0;
/** Damping applied to extrapolated earner log-growth beyond the last read.
 *  Set to 1.0 (no extra damping): the log-time clock already saturates
 *  growth — x only rises 25% from the deadline to the target — and the
 *  synthetic cases score better undamped (0.218) than at 0.75 (0.249). */
const DAMP = 1.0;
/** Shrinkage strength: a trophy's slope weight is m / (m + SHRINK_K), where m
 *  is its number of informative reads; the rest of the weight goes to the
 *  cross-trophy median slope.
 *  Set to 0 (no shrinkage): on both synthetic cases shrinkage only adds
 *  bias — true slopes genuinely differ per trophy (0.4 to 1.8) and the
 *  per-trophy fits are good enough — and it gets monotonically worse as
 *  K grows (clean 0.218 / 0.334 at K=0/3; noisy 0.251 / 0.322). */
const SHRINK_K = 0;
/** Largest log-growth of players allowed beyond the last read (about 55x). */
const MAX_PLAYER_LOG_GROWTH = 4.0;
/** Floor inside logs for a zero earner count. */
const EARNS_FLOOR = 0.5;
/** Spread for a trophy already at (essentially) 100%: it stays there. */
const CERTAIN_SPREAD = 0.2;

// ---------------------------------------------------------------------------
// Small numeric helpers.
// ---------------------------------------------------------------------------

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The game's release instant; the clock all fits run on. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

/** A percent clamped to the range the scorer's logit is defined on. */
const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
/** logit of a percent. */
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
/** Inverse logit, back to a percent. */
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** A number rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** ln(1 + hours) from an ISO instant to the release instant. */
const logTime = (iso: string): number =>
  Math.log1p(Math.max(-0.999, (Date.parse(iso) - RELEASE_MS) / 3_600_000));

interface HistoryRead {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface TrophyMeta {
  id: number;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophyMeta[];
  history: HistoryRead[];
}

/** Least-squares line through (x, y) points, with its residual RMS. */
function ols(xs: number[], ys: number[]): { slope: number; intercept: number; rmse: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    ss += r * r;
  }
  return { slope, intercept, rmse: Math.sqrt(ss / n) };
}

/** Median of a list of numbers (0 for an empty list). */
function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Collapse runs of consecutive identical reads to their first read.
 *  While Exophase has not refreshed, the page repeats verbatim; those
 *  repeats are one observation of a frozen page, not many observations of a
 *  trend. Feeding them all to a line fit drags every slope toward zero —
 *  the eight identical launch-night reads would halve the fitted growth. */
function dedupReads(reads: HistoryRead[]): HistoryRead[] {
  const out: HistoryRead[] = [];
  for (const r of reads) {
    const prev = out[out.length - 1];
    if (
      prev !== undefined &&
      prev.players_tracked === r.players_tracked &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    out.push(r);
  }
  return out;
}

/** Is the history flat — every read the same players and the same rates? */
function isFlatHistory(reads: HistoryRead[]): boolean {
  if (reads.length < 3) return true;
  const first = reads[0];
  for (let j = 1; j < reads.length; j++) {
    const r = reads[j];
    if (r.players_tracked !== first.players_tracked) return false;
    if (r.rates.length !== first.rates.length) return false;
    for (let i = 0; i < r.rates.length; i++) {
      if (r.rates[i] !== first.rates[i]) return false;
    }
  }
  return true;
}

/** The last non-null rate for one trophy, searching back through the reads. */
function lastKnownRate(reads: HistoryRead[], i: number): number | null {
  for (let j = reads.length - 1; j >= 0; j--) {
    const r = reads[j].rates[i];
    if (r !== null && r !== undefined) return r;
  }
  return null;
}

/** One trophy's (log-time, log-earners) points, its last earners, and how many
 *  reads actually carried a positive earner count. */
function trophyPoints(
  reads: HistoryRead[],
  i: number,
  lastP: number,
): { xs: number[]; ys: number[]; informative: number; lastE: number } {
  const xs: number[] = [];
  const ys: number[] = [];
  let informative = 0;
  let lastE = EARNS_FLOOR;
  const last = reads[reads.length - 1];
  for (const r of reads) {
    const rate = r.rates[i];
    if (rate === null || rate === undefined) continue;
    const p = r.players_tracked > 0 ? r.players_tracked : lastP;
    const e = Math.max((rate / 100) * p, EARNS_FLOOR);
    xs.push(logTime(r.at));
    ys.push(Math.log(e));
    if (e > EARNS_FLOOR) informative++;
    if (r === last) lastE = e;
  }
  if (last.rates[i] === null || last.rates[i] === undefined) {
    for (let j = reads.length - 1; j >= 0; j--) {
      const rate = reads[j].rates[i];
      if (rate !== null && rate !== undefined) {
        lastE = Math.max((rate / 100) * (reads[j].players_tracked || lastP), EARNS_FLOOR);
        break;
      }
    }
  }
  return { xs, ys, informative, lastE };
}

/** One trophy's forecast row. */
function forecastTrophy(
  reads: HistoryRead[],
  i: number,
  trophyId: number,
  xTarget: number,
  horizonDays: number,
  playersFit: { slope: number; intercept: number; rmse: number } | null,
  medianSlope: number,
  medianRmse: number,
): { id: number; q: number[] } {
  const last = reads[reads.length - 1];
  const lastRate = lastKnownRate(reads, i) ?? 1.0;
  const lastP = last.players_tracked > 0 ? last.players_tracked : 1;

  // A trophy (essentially) everyone already holds stays held.
  if (lastRate >= 99.99) {
    const l = logit(99.99);
    const q = Z.map((z) => round4(clampPct(invLogit(l + z * CERTAIN_SPREAD))));
    return { id: trophyId, q: enforceOrder(q) };
  }

  // This trophy's (log-time, log-earners) points.
  const { xs, ys, informative, lastE } = trophyPoints(reads, i, lastP);

  const fit = ols(xs, ys);
  let logETarget = Math.log(lastE);
  let rmse = medianRmse;
  if (fit !== null && xs.length >= 2) {
    rmse = fit.rmse;
    // Shrink this trophy's slope toward the cross-trophy median slope.
    const w = informative / (informative + SHRINK_K);
    const slope = w * fit.slope + (1 - w) * medianSlope;
    const raw = fit.intercept + slope * xTarget;
    // Earners are cumulative: they never fall, and growth beyond the last
    // read is damped, because progress saturates.
    const growth = raw - Math.log(lastE);
    logETarget = growth > 0 ? Math.log(lastE) + DAMP * growth : Math.log(lastE);
  }

  // Players at the target: the fitted line, capped so it cannot explode,
  // and never below the last read (players tracked do not shrink).
  let logPTarget = Math.log(lastP);
  if (playersFit !== null) {
    const rawP = playersFit.intercept + playersFit.slope * xTarget;
    logPTarget = Math.min(rawP, Math.log(lastP) + MAX_PLAYER_LOG_GROWTH);
    logPTarget = Math.max(logPTarget, Math.log(lastP));
  }
  const pTarget = Math.exp(logPTarget);
  const eTarget = Math.min(Math.exp(logETarget), pTarget);
  const rMed = clampPct((100 * eTarget) / pTarget);

  // Spread: the base random walk over the horizon, plus this trophy's fit
  // residual (log-earners) and the players fit's residual, combined.
  const base = SIGMA0 * Math.sqrt(horizonDays);
  const fitSigma = Math.sqrt(rmse * rmse + (playersFit?.rmse ?? 0) ** 2);
  const sigma = Math.sqrt(base * base + Math.min(fitSigma, 2) ** 2);
  const l = logit(rMed);
  const q = Z.map((z) => round4(clampPct(invLogit(l + z * sigma))));
  return { id: trophyId, q: enforceOrder(q) };
}

/** Force a quantile row non-decreasing (rounding can only tie, never invert). */
function enforceOrder(q: number[]): number[] {
  const out = [...q];
  for (let j = 1; j < out.length; j++) {
    if (out[j] < out[j - 1]) out[j] = out[j - 1];
  }
  return out;
}

/** Forecast every trophy of one case. */
function forecastCase(c: CaseFile): { trophies: Array<{ id: number; q: number[] }> } {
  const reads = c.history;
  const horizonDays = c.horizon_minutes / 1440;
  const targetIso = c.target || new Date(Date.parse(c.at) + c.horizon_minutes * 60_000).toISOString();
  const xTarget = logTime(targetIso);

  // No history at all: emit a valid shape (never crash the run).
  if (reads.length === 0) {
    return {
      trophies: c.trophies.map((t) => ({
        id: t.id,
        q: Z.map((z) => round4(clampPct(invLogit(logit(1) + z * SIGMA0 * Math.sqrt(horizonDays))))),
      })),
    };
  }

  // Flat history: no trend can be read from it. Median = last rate, spread
  // the base random walk — persistence — but with the flat case's wider
  // sigma, because a frozen page can jump a long way at its next refresh.
  if (isFlatHistory(reads)) {
    const spread = FLAT_SIGMA0 * Math.sqrt(horizonDays);
    return {
      trophies: c.trophies.map((t, i) => {
        const lastRate = lastKnownRate(reads, i) ?? 1.0;
        if (lastRate >= 99.99) {
          const l = logit(99.99);
          return { id: t.id, q: enforceOrder(Z.map((z) => round4(clampPct(invLogit(l + z * CERTAIN_SPREAD))))) };
        }
        const l = logit(lastRate);
        return { id: t.id, q: enforceOrder(Z.map((z) => round4(clampPct(invLogit(l + z * spread))))) };
      }),
    };
  }

  // Non-flat history: fit on the distinct page states, not on every
  // repeated read of a frozen page (see dedupReads). And drop the first
  // state: its values are the page as found when recording started — nobody
  // knows how long the page had already shown them — so its timestamp does
  // not date them, and as the farthest point in log-time it would lever
  // every fitted slope toward zero. The remaining states are all dated by
  // the refresh that first showed them.
  const deduped = dedupReads(reads);
  const fitReads = deduped.length >= 3 ? deduped.slice(1) : deduped;

  // Players fit: log players against log time, over all reads.
  const lastP = fitReads[fitReads.length - 1].players_tracked > 0
    ? fitReads[fitReads.length - 1].players_tracked
    : 1;
  const pxs: number[] = [];
  const pys: number[] = [];
  for (const r of fitReads) {
    if (r.players_tracked > 0) {
      pxs.push(logTime(r.at));
      pys.push(Math.log(r.players_tracked));
    }
  }
  const playersFit = ols(pxs, pys);

  // First pass over trophies: their slopes and residual RMS, for the
  // cross-trophy median that the shrinkage and the default spread use.
  const slopes: number[] = [];
  const rmses: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const { xs, ys, informative } = trophyPoints(fitReads, i, lastP);
    const fit = ols(xs, ys);
    if (fit !== null && informative >= 3) {
      slopes.push(fit.slope);
      rmses.push(fit.rmse);
    }
  }
  const medianSlope = median(slopes);
  const medianRmse = median(rmses);

  return {
    trophies: c.trophies.map((t, i) =>
      forecastTrophy(fitReads, i, t.id, xTarget, horizonDays, playersFit, medianSlope, medianRmse),
    ),
  };
}

// ---------------------------------------------------------------------------
// Entry point: read every <id>.in in the input directory, write <id>.out.
// ---------------------------------------------------------------------------

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history === undefined) throw new Error(`${file} has no history`);
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
