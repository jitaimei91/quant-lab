"""Pure-NumPy Gaussian HMM with EM training for regime detection.

Implements a Hidden Markov Model with diagonal Gaussian emissions.
Uses the Baum-Welch (EM) algorithm for training, Viterbi for decoding,
and forward-backward for posterior inference.

Numerical stability: all computations in log space using logsumexp.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HMMState:
    n_states: int
    means: np.ndarray          # (n_states, n_features)
    covariances: np.ndarray    # (n_states, n_features, n_features) -- diagonal cov
    transition: np.ndarray     # (n_states, n_states)
    initial: np.ndarray        # (n_states,)


def _log_gaussian_diag(x: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Log probability of observations under diagonal Gaussian emissions.

    Parameters
    ----------
    x : (T, n_features)
    means : (n_states, n_features)
    covs : (n_states, n_features, n_features)  diagonal -- only diagonal used

    Returns
    -------
    (T, n_states) log likelihoods
    """
    T, n_features = x.shape
    n_states = means.shape[0]
    log_probs = np.zeros((T, n_states))

    for k in range(n_states):
        mu = means[k]  # (n_features,)
        # Extract diagonal variances
        var = np.diag(covs[k])  # (n_features,)
        var = np.maximum(var, 1e-6)  # numerical floor

        diff = x - mu  # (T, n_features)
        log_det = np.sum(np.log(var))
        mahal = np.sum(diff ** 2 / var, axis=1)  # (T,)
        log_probs[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)

    return log_probs


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable logsumexp."""
    if axis is None:
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))
    a_max = np.max(a, axis=axis, keepdims=True)
    result = a_max.squeeze(axis=axis) + np.log(
        np.sum(np.exp(a - a_max), axis=axis)
    )
    return result


def _forward(log_emit: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray) -> np.ndarray:
    """Forward algorithm in log space.

    Parameters
    ----------
    log_emit : (T, n_states)
    log_trans : (n_states, n_states)  -- log_trans[i,j] = log P(j | i)
    log_init  : (n_states,)

    Returns
    -------
    alpha : (T, n_states)  log forward variables
    """
    T, n_states = log_emit.shape
    alpha = np.full((T, n_states), -np.inf)
    alpha[0] = log_init + log_emit[0]

    for t in range(1, T):
        # alpha[t, j] = log_emit[t,j] + logsumexp_i(alpha[t-1,i] + log_trans[i,j])
        for j in range(n_states):
            alpha[t, j] = log_emit[t, j] + _logsumexp(alpha[t - 1] + log_trans[:, j])

    return alpha


def _backward(log_emit: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Backward algorithm in log space.

    Returns
    -------
    beta : (T, n_states)  log backward variables
    """
    T, n_states = log_emit.shape
    beta = np.zeros((T, n_states))  # log(1) = 0 at T-1

    for t in range(T - 2, -1, -1):
        for i in range(n_states):
            beta[t, i] = _logsumexp(log_trans[i] + log_emit[t + 1] + beta[t + 1])

    return beta


def fit_hmm(
    observations: np.ndarray,
    n_states: int = 4,
    n_iter: int = 50,
    seed: int = 42,
) -> HMMState:
    """EM (Baum-Welch) training for diagonal-covariance Gaussian HMM.

    Parameters
    ----------
    observations : (T, n_features)  -- raw observations
    n_states : number of hidden states
    n_iter : number of EM iterations
    seed : random seed for initialization

    Returns
    -------
    Trained HMMState
    """
    rng = np.random.default_rng(seed)
    T, n_features = observations.shape

    # --- Initialization ---
    # K-means style: random subset of observations as initial means
    idx = rng.choice(T, size=n_states, replace=False)
    means = observations[idx].copy().astype(float)

    # Diagonal covariances initialized to identity
    covs = np.array([np.diag(np.var(observations, axis=0) + 1e-3) for _ in range(n_states)])

    # Transition matrix: uniform
    transition = np.ones((n_states, n_states)) / n_states

    # Initial distribution: uniform
    initial = np.ones(n_states) / n_states

    prev_log_lik = -np.inf

    for _iter in range(n_iter):
        log_trans = np.log(transition + 1e-300)
        log_init = np.log(initial + 1e-300)

        # --- E-step ---
        log_emit = _log_gaussian_diag(observations, means, covs)
        alpha = _forward(log_emit, log_trans, log_init)
        beta = _backward(log_emit, log_trans)

        # Log-likelihood
        log_lik = _logsumexp(alpha[-1])

        # Posteriors: gamma[t, k] = P(state_t=k | obs)
        log_gamma = alpha + beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, np.newaxis]
        gamma = np.exp(log_gamma)

        # Xi: xi[t, i, j] = P(state_t=i, state_{t+1}=j | obs)
        # Shape: (T-1, n_states, n_states)
        xi = np.zeros((T - 1, n_states, n_states))
        for t in range(T - 1):
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = (
                        alpha[t, i]
                        + log_trans[i, j]
                        + log_emit[t + 1, j]
                        + beta[t + 1, j]
                    )
            # Normalize
            xi[t] = np.exp(xi[t] - _logsumexp(xi[t].ravel()))

        # --- M-step ---
        # Update initial distribution
        initial = gamma[0] + 1e-10
        initial /= initial.sum()

        # Update transition matrix
        xi_sum = xi.sum(axis=0)  # (n_states, n_states)
        row_sum = xi_sum.sum(axis=1, keepdims=True)
        transition = (xi_sum + 1e-10) / (row_sum + 1e-10 * n_states)

        # Update means and (diagonal) covariances
        gamma_sum = gamma.sum(axis=0)  # (n_states,)
        for k in range(n_states):
            w = gamma[:, k]  # (T,)
            wsum = gamma_sum[k] + 1e-10
            means[k] = (w[:, np.newaxis] * observations).sum(axis=0) / wsum
            diff = observations - means[k]
            var_k = (w[:, np.newaxis] * diff ** 2).sum(axis=0) / wsum
            var_k = np.maximum(var_k, 1e-6)
            covs[k] = np.diag(var_k)

        # Convergence check
        if abs(log_lik - prev_log_lik) < 1e-4:
            break
        prev_log_lik = log_lik

    return HMMState(
        n_states=n_states,
        means=means,
        covariances=covs,
        transition=transition,
        initial=initial,
    )


def viterbi(observations: np.ndarray, hmm: HMMState) -> np.ndarray:
    """Most likely hidden-state sequence via Viterbi algorithm.

    Parameters
    ----------
    observations : (T, n_features)
    hmm : trained HMMState

    Returns
    -------
    states : (T,) integer array of most likely state indices
    """
    T = observations.shape[0]
    n_states = hmm.n_states

    log_emit = _log_gaussian_diag(observations, hmm.means, hmm.covariances)
    log_trans = np.log(hmm.transition + 1e-300)
    log_init = np.log(hmm.initial + 1e-300)

    viterbi_mat = np.full((T, n_states), -np.inf)
    backptr = np.zeros((T, n_states), dtype=int)

    viterbi_mat[0] = log_init + log_emit[0]

    for t in range(1, T):
        for j in range(n_states):
            scores = viterbi_mat[t - 1] + log_trans[:, j]
            backptr[t, j] = int(np.argmax(scores))
            viterbi_mat[t, j] = scores[backptr[t, j]] + log_emit[t, j]

    # Backtrack
    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(viterbi_mat[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]

    return states


def forward_backward(observations: np.ndarray, hmm: HMMState) -> np.ndarray:
    """Posterior state probabilities via forward-backward algorithm.

    Parameters
    ----------
    observations : (T, n_features)
    hmm : trained HMMState

    Returns
    -------
    posteriors : (T, n_states)  -- each row sums to 1
    """
    log_emit = _log_gaussian_diag(observations, hmm.means, hmm.covariances)
    log_trans = np.log(hmm.transition + 1e-300)
    log_init = np.log(hmm.initial + 1e-300)

    alpha = _forward(log_emit, log_trans, log_init)
    beta = _backward(log_emit, log_trans)

    log_gamma = alpha + beta
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, np.newaxis]
    return np.exp(log_gamma)


def save_hmm(hmm: HMMState, path: Path) -> None:
    """Serialize HMMState to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "n_states": hmm.n_states,
        "means": hmm.means.tolist(),
        "covariances": hmm.covariances.tolist(),
        "transition": hmm.transition.tolist(),
        "initial": hmm.initial.tolist(),
    }
    path.write_text(json.dumps(data))


def load_hmm(path: Path) -> HMMState | None:
    """Deserialize HMMState from JSON. Returns None if file missing or invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return HMMState(
            n_states=data["n_states"],
            means=np.array(data["means"]),
            covariances=np.array(data["covariances"]),
            transition=np.array(data["transition"]),
            initial=np.array(data["initial"]),
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
