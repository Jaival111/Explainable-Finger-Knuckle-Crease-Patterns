"""Uniqueness of 2D finger knuckle patterns (Sec. 3.4, 4.3).

1. Templates T = {(k_i, f_i)} with m keypoints are extracted by KnuckleCreasePoint.
2. Tolerances Delta_k / Delta_f are estimated from genuine correspondences
   (KnucklePointPair matches between images of the same finger) as the 97.5%
   quantile of the location / descriptor distances.
3. A joint mixture model (Eq. 18) - each component a bivariate Gaussian on the
   location times a 256-d Gaussian on the descriptor - is fitted by EM; the
   number of components N* is chosen by BIC.
4. Each template gets its own mixture density f_T (shared components, weights
   = mean responsibilities of the template's keypoints). The random keypoint
   match probability p(Tp, Tg) of Eq. (20) is evaluated as
       p(Tp, Tg) = sum_ab  w_a^p w_b^g  P(|k_a-k_b| <= Dk) P(|f_a-f_b| <= Df)
   with the per-component-pair probabilities estimated by Monte Carlo.
5. lambda = m n p(Tp, Tg);  P(w; Tp, Tg) = P(Poisson(lambda) >= w)   (Eq. 19)
   and the trimmed mean over impostor template pairs gives FRC_alpha (Eq. 22).

Usage:
    python uniqueness.py [--delta-k 6.75 --delta-f 0.726]
"""
import argparse
import csv
import json

import numpy as np
import torch
from scipy.linalg import cho_factor, solve_triangular
from scipy.special import logsumexp
from scipy.stats import poisson
from sklearn.cluster import KMeans
from tqdm import tqdm

import config
from common import get_device, seed_everything
from dataset import all_ordered_pairs, list_samples, load_images
from pipeline import extract_features, index_features, load_trained, match_batch


# ----------------------------------------------------------------------------
# Joint location-feature mixture model (Eq. 18)
# ----------------------------------------------------------------------------
def _chol(cov):
    return np.tril(cho_factor(cov, lower=True)[0])


def _log_gauss(X, mean, chol):
    sol = solve_triangular(chol, (X - mean).T, lower=True)
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    return -0.5 * (X.shape[1] * np.log(2 * np.pi) + logdet + (sol ** 2).sum(0))


class JointKeypointGMM:
    def __init__(self, n_components, reg_k=1e-2, reg_f=1e-4, max_iter=100, tol=1e-4, seed=0):
        self.n_components = n_components
        self.reg_k, self.reg_f = reg_k, reg_f
        self.max_iter, self.tol, self.seed = max_iter, tol, seed

    def _m_step(self, K, F, resp):
        nk = resp.sum(0) + 1e-10
        self.weights = nk / nk.sum()
        self.mu_k = resp.T @ K / nk[:, None]
        self.mu_f = resp.T @ F / nk[:, None]
        self.chol_k, self.chol_f = [], []
        for c in range(self.n_components):
            dk, df = K - self.mu_k[c], F - self.mu_f[c]
            cov_k = (resp[:, c, None] * dk).T @ dk / nk[c] + self.reg_k * np.eye(K.shape[1])
            cov_f = (resp[:, c, None] * df).T @ df / nk[c] + self.reg_f * np.eye(F.shape[1])
            self.chol_k.append(_chol(cov_k))
            self.chol_f.append(_chol(cov_f))

    def _log_prob(self, K, F):
        out = np.empty((len(K), self.n_components))
        for c in range(self.n_components):
            out[:, c] = (np.log(self.weights[c]) + _log_gauss(K, self.mu_k[c], self.chol_k[c])
                         + _log_gauss(F, self.mu_f[c], self.chol_f[c]))
        return out

    def fit(self, K, F):
        K, F = K.astype(np.float64), F.astype(np.float64)
        X0 = np.hstack([K / K.std(), F / F.std()])
        labels = KMeans(self.n_components, n_init=5, random_state=self.seed).fit_predict(X0)
        self._m_step(K, F, np.eye(self.n_components)[labels])
        prev = -np.inf
        for _ in range(self.max_iter):
            logp = self._log_prob(K, F)
            lse = logsumexp(logp, axis=1)
            self._m_step(K, F, np.exp(logp - lse[:, None]))
            ll = lse.mean()
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.total_log_likelihood = logsumexp(self._log_prob(K, F), axis=1).sum()
        return self

    def n_parameters(self, dk, df):
        per = dk + dk * (dk + 1) / 2 + df + df * (df + 1) / 2
        return self.n_components - 1 + self.n_components * per

    def bic(self, K, F):
        return -2 * self.total_log_likelihood + self.n_parameters(K.shape[1], F.shape[1]) * np.log(len(K))

    def responsibilities(self, K, F):
        logp = self._log_prob(K.astype(np.float64), F.astype(np.float64))
        return np.exp(logp - logsumexp(logp, axis=1, keepdims=True))

    def sample(self, c, n, rng):
        zk = rng.standard_normal((n, len(self.mu_k[c])))
        zf = rng.standard_normal((n, len(self.mu_f[c])))
        return self.mu_k[c] + zk @ self.chol_k[c].T, self.mu_f[c] + zf @ self.chol_f[c].T


def fit_with_bic(K, F, max_components, seed):
    best, best_bic = None, np.inf
    for n in range(1, max_components + 1):
        try:
            gmm = JointKeypointGMM(n, seed=seed).fit(K, F)
        except np.linalg.LinAlgError:
            print(f"  N={n}: covariance not positive definite, skipped")
            continue
        bic = gmm.bic(K, F)
        print(f"  N={n}: BIC={bic:.1f}")
        if bic < best_bic:
            best, best_bic = gmm, bic
    return best


def component_match_probs(gmm, dks, dfs, n_samples, rng, chunk=20000):
    """P(|k_a - k_b| <= Dk) and P(|f_a - f_b| <= Df) for every component pair (a, b)
    and every threshold. Sampled descriptors are L2-normalised like real ones."""
    N = gmm.n_components
    pk = np.zeros((N, N, len(dks)))
    pf = np.zeros((N, N, len(dfs)))
    dks, dfs = np.asarray(dks), np.asarray(dfs)
    for a in range(N):
        for b in range(N):
            done = 0
            while done < n_samples:
                s = min(chunk, n_samples - done)
                ka, fa = gmm.sample(a, s, rng)
                kb, fb = gmm.sample(b, s, rng)
                fa /= np.linalg.norm(fa, axis=1, keepdims=True) + 1e-12
                fb /= np.linalg.norm(fb, axis=1, keepdims=True) + 1e-12
                dist_k = np.linalg.norm(ka - kb, axis=1)
                dist_f = np.linalg.norm(fa - fb, axis=1)
                pk[a, b] += (dist_k[:, None] <= dks[None]).sum(0)
                pf[a, b] += (dist_f[:, None] <= dfs[None]).sum(0)
                done += s
    return pk / n_samples, pf / n_samples


# ----------------------------------------------------------------------------
# Tolerance estimation from genuine correspondences
# ----------------------------------------------------------------------------
@torch.no_grad()
def estimate_tolerances(matcher, feats, subjects, device, conf, quantile, batch_size=64):
    ii, jj, labels = all_ordered_pairs(subjects)
    gen = labels == 1
    ii, jj = ii[gen], jj[gen]
    dk_all, df_all = [], []
    for s in tqdm(range(0, len(ii), batch_size), desc="genuine correspondences"):
        bi = torch.as_tensor(ii[s:s + batch_size], device=device)
        bj = torch.as_tensor(jj[s:s + batch_size], device=device)
        f0, f1 = index_features(feats, bi), index_features(feats, bj)
        res = match_batch(matcher, f0, f1, config.NUM_CORR, (config.IMG_H, config.IMG_W), conf)
        m0 = res["matches0"]
        for b in range(len(bi)):
            i = torch.nonzero(m0[b] >= 0).squeeze(1)
            j = m0[b, i]
            dk_all.append((f0["keypoints"][b, i] - f1["keypoints"][b, j]).norm(dim=1).cpu().numpy())
            df_all.append((f0["descriptors"][b, i] - f1["descriptors"][b, j]).norm(dim=1).cpu().numpy())
    dk_all, df_all = np.concatenate(dk_all), np.concatenate(df_all)
    if len(dk_all) == 0:
        raise RuntimeError("No genuine correspondences found; pass --delta-k / --delta-f explicitly.")
    return float(np.quantile(dk_all, quantile)), float(np.quantile(df_all, quantile)), len(dk_all)


# ----------------------------------------------------------------------------
# FRC (Eq. 19, 21, 22)
# ----------------------------------------------------------------------------
def frc_statistics(p_pairs, m, n, w, alpha):
    lam = m * n * p_pairs
    pw = poisson.sf(w - 1, lam)                      # P(at least w matches)
    lo, hi = np.quantile(pw, [alpha / 2, 1 - alpha / 2])
    keep = (pw >= lo) & (pw <= hi)
    return {"p_mean": float(p_pairs.mean()), "lambda_alpha": float(lam[keep].mean()),
            "frc": float(pw.mean()), "frc_alpha": float(pw[keep].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-k", type=float, default=None, help="location tolerance (px)")
    ap.add_argument("--delta-f", type=float, default=None, help="descriptor tolerance")
    ap.add_argument("--m", type=int, default=config.UNI_TEMPLATE_SIZE)
    ap.add_argument("--w", type=int, default=config.UNI_W)
    ap.add_argument("--alpha", type=float, default=config.UNI_ALPHA)
    ap.add_argument("--max-components", type=int, default=config.UNI_MAX_COMPONENTS)
    ap.add_argument("--mc-samples", type=int, default=config.UNI_MC_SAMPLES)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    rng = np.random.default_rng(config.SEED)
    detector, matcher = load_trained(device, cgn=False)

    samples = list_samples()
    images = load_images(samples)
    subjects = np.array([s.subject for s in samples])

    # -- tolerances ------------------------------------------------------------
    dk, df = args.delta_k, args.delta_f
    if dk is None or df is None:
        feats_full = extract_features(detector, images, device)
        est_k, est_f, n_corr = estimate_tolerances(matcher, feats_full, subjects, device,
                                                   config.UNI_GENUINE_CONF, config.UNI_DELTA_QUANTILE)
        print(f"Estimated from {n_corr} genuine correspondences: Delta_k={est_k:.3f}, Delta_f={est_f:.3f}")
        dk = est_k if dk is None else dk
        df = est_f if df is None else df

    # -- templates & mixture model ---------------------------------------------
    feats = extract_features(detector, images, device, max_keypoints=args.m)
    K = feats["keypoints"].cpu().numpy()            # (T, m, 2)
    F = feats["descriptors"].cpu().numpy()          # (T, m, D)
    T = len(K)
    print(f"Fitting joint mixture model on {T} templates x {args.m} keypoints")
    gmm = fit_with_bic(K.reshape(-1, 2), F.reshape(-1, F.shape[-1]), args.max_components, config.SEED)
    print(f"Selected N* = {gmm.n_components} components")

    W = gmm.responsibilities(K.reshape(-1, 2), F.reshape(-1, F.shape[-1])).reshape(T, args.m, -1).mean(1)

    # -- parameter grid in the spirit of Table 5 -------------------------------
    dk_grid = sorted({round(dk * s, 3) for s in (3.75 / 6.75, 5.25 / 6.75, 1.0, 8.25 / 6.75, 9.75 / 6.75)})
    df_grid = sorted({round(df + o, 3) for o in (-0.1, -0.05, 0.0, 0.05, 0.1) if df + o > 0})
    w_grid = [args.w - 2, args.w, args.w + 2, args.w + 4]
    print("Monte-Carlo estimation of component match probabilities ...")
    pk, pf = component_match_probs(gmm, dk_grid, df_grid, args.mc_samples, rng)

    iu, ju = np.triu_indices(T, k=1)
    impostor = subjects[iu] != subjects[ju]
    iu, ju = iu[impostor], ju[impostor]

    def p_pairs(a, b):
        P = pk[:, :, a] * pf[:, :, b]
        return np.einsum("ta,ab,tb->t", W[iu], P, W[ju])   # Eq. (20) per impostor pair

    ik, jf = dk_grid.index(round(dk, 3)), df_grid.index(round(df, 3))
    rows = []
    for w in w_grid:
        rows.append((dk_grid[ik], df_grid[jf], w, frc_statistics(p_pairs(ik, jf), args.m, args.m, w, args.alpha)))
    for a, d in enumerate(dk_grid):
        if a != ik:
            rows.append((d, df_grid[jf], args.w, frc_statistics(p_pairs(a, jf), args.m, args.m, args.w, args.alpha)))
    for b, d in enumerate(df_grid):
        if b != jf:
            rows.append((dk_grid[ik], d, args.w, frc_statistics(p_pairs(ik, b), args.m, args.m, args.w, args.alpha)))

    print(f"\n{len(iu)} impostor template pairs, m = n = {args.m}, alpha = {args.alpha}")
    print(f"{'Dk':>7} {'Df':>7} {'(m,n,w,a)':>18} {'p(Tp,Tg)':>11} {'lambda_a':>9} {'FRC_a':>11}")
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "uniqueness.csv", "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["delta_k", "delta_f", "m", "n", "w", "alpha", "p_mean", "lambda_alpha", "frc", "frc_alpha"])
        for d_k, d_f, w, r in rows:
            print(f"{d_k:7.3f} {d_f:7.3f} {f'({args.m},{args.m},{w},{args.alpha})':>18} "
                  f"{r['p_mean']:11.3e} {r['lambda_alpha']:9.3f} {r['frc_alpha']:11.3e}")
            wr.writerow([d_k, d_f, args.m, args.m, w, args.alpha, r["p_mean"], r["lambda_alpha"],
                         r["frc"], r["frc_alpha"]])
    with open(out_dir / "uniqueness_model.json", "w") as fj:
        json.dump({"delta_k": dk, "delta_f": df, "n_components": gmm.n_components,
                   "weights": gmm.weights.tolist()}, fj, indent=2)
    print(f"Saved results to {out_dir / 'uniqueness.csv'}")


if __name__ == "__main__":
    main()
