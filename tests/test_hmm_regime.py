"""Tests for engine/hmm_regime.py — Gaussian HMM with EM training."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from quant_lab.engine.hmm_regime import (
    HMMState,
    fit_hmm,
    forward_backward,
    load_hmm,
    save_hmm,
    viterbi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_2regime_data(T: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 2-regime observations with clear separation.

    Regime 0: low mean (5, 0.01), Regime 1: high mean (25, 0.05)
    Returns (observations (T,2), true_states (T,))
    """
    rng = np.random.default_rng(seed)
    # Slow-switching: regime changes with prob 0.05 per step
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        if states[t - 1] == 0:
            states[t] = int(rng.random() < 0.05)
        else:
            states[t] = int(rng.random() > 0.05)

    obs = np.zeros((T, 2))
    for t in range(T):
        if states[t] == 0:
            obs[t] = rng.normal([5.0, 0.01], [0.5, 0.005])
        else:
            obs[t] = rng.normal([25.0, 0.05], [1.5, 0.01])

    return obs, states


def _make_hmm_state(n_states: int = 2, n_features: int = 2) -> HMMState:
    """Build a simple HMMState for testing."""
    means = np.array([[5.0, 0.01], [25.0, 0.05]])
    covs = np.array([np.diag([0.25, 2.5e-5]), np.diag([2.25, 1e-4])])
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    initial = np.array([0.5, 0.5])
    return HMMState(
        n_states=n_states,
        means=means,
        covariances=covs,
        transition=transition,
        initial=initial,
    )


# ---------------------------------------------------------------------------
# EM training: recovered means within 30%
# ---------------------------------------------------------------------------

def test_fit_hmm_recovers_means():
    """EM on 2-regime synthetic data should recover state means within 30%."""
    obs, _ = _make_2regime_data(T=300, seed=42)
    hmm = fit_hmm(obs, n_states=2, n_iter=50, seed=42)

    # Sort fitted means by first feature to match regime ordering
    fitted_means = hmm.means[np.argsort(hmm.means[:, 0])]

    true_means = np.array([[5.0, 0.01], [25.0, 0.05]])
    for k in range(2):
        for f in range(2):
            true_val = true_means[k, f]
            fitted_val = fitted_means[k, f]
            if abs(true_val) > 1e-8:
                rel_err = abs(fitted_val - true_val) / abs(true_val)
                assert rel_err < 0.30, (
                    f"State {k} feature {f}: fitted={fitted_val:.4f} vs true={true_val:.4f}, "
                    f"rel_err={rel_err:.2%} > 30%"
                )


def test_fit_hmm_shapes():
    """fit_hmm returns correctly shaped arrays."""
    obs = np.random.default_rng(0).normal(size=(100, 5))
    hmm = fit_hmm(obs, n_states=3, n_iter=10, seed=0)

    assert hmm.n_states == 3
    assert hmm.means.shape == (3, 5)
    assert hmm.covariances.shape == (3, 5, 5)
    assert hmm.transition.shape == (3, 3)
    assert hmm.initial.shape == (3,)


def test_fit_hmm_transition_rows_sum_to_1():
    """Transition matrix rows should sum to 1."""
    obs = np.random.default_rng(1).normal(size=(80, 3))
    hmm = fit_hmm(obs, n_states=2, n_iter=20, seed=1)
    np.testing.assert_allclose(hmm.transition.sum(axis=1), np.ones(2), atol=1e-6)


def test_fit_hmm_initial_sums_to_1():
    """Initial distribution should sum to 1."""
    obs = np.random.default_rng(2).normal(size=(80, 3))
    hmm = fit_hmm(obs, n_states=2, n_iter=20, seed=2)
    np.testing.assert_allclose(hmm.initial.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Viterbi: sensible state sequence for clear 2-regime data
# ---------------------------------------------------------------------------

def test_viterbi_output_shape():
    """viterbi returns integer array of shape (T,)."""
    obs, _ = _make_2regime_data(T=100)
    hmm = _make_hmm_state()
    states = viterbi(obs, hmm)

    assert states.shape == (100,)
    assert states.dtype == int or np.issubdtype(states.dtype, np.integer)


def test_viterbi_state_values_in_range():
    """viterbi output values are in [0, n_states)."""
    obs, _ = _make_2regime_data(T=100)
    hmm = _make_hmm_state()
    states = viterbi(obs, hmm)

    assert np.all(states >= 0)
    assert np.all(states < hmm.n_states)


def test_viterbi_separates_regimes():
    """Viterbi on well-separated 2-regime data should track regimes reasonably."""
    obs, true_states = _make_2regime_data(T=300, seed=7)
    hmm = _make_hmm_state()
    pred = viterbi(obs, hmm)

    # Compute accuracy (allowing label flip: try both assignments)
    acc_direct = np.mean(pred == true_states)
    acc_flipped = np.mean((1 - pred) == true_states)
    accuracy = max(acc_direct, acc_flipped)

    # Should correctly identify regimes >80% of the time
    assert accuracy > 0.80, f"Viterbi accuracy {accuracy:.2%} < 80%"


# ---------------------------------------------------------------------------
# forward_backward: posteriors sum to 1
# ---------------------------------------------------------------------------

def test_forward_backward_output_shape():
    """forward_backward returns (T, n_states) array."""
    obs, _ = _make_2regime_data(T=50)
    hmm = _make_hmm_state()
    posteriors = forward_backward(obs, hmm)

    assert posteriors.shape == (50, 2)


def test_forward_backward_posteriors_sum_to_1():
    """Posteriors must sum to 1 across states at each timestep."""
    obs, _ = _make_2regime_data(T=100)
    hmm = _make_hmm_state()
    posteriors = forward_backward(obs, hmm)

    row_sums = posteriors.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(100), atol=1e-6)


def test_forward_backward_posteriors_nonnegative():
    """Posteriors must be non-negative."""
    obs, _ = _make_2regime_data(T=100)
    hmm = _make_hmm_state()
    posteriors = forward_backward(obs, hmm)

    assert np.all(posteriors >= 0)


def test_forward_backward_4state_shapes():
    """4-state HMM posteriors shape and sum."""
    obs = np.random.default_rng(10).normal(size=(60, 5))
    hmm = fit_hmm(obs, n_states=4, n_iter=5, seed=10)
    posteriors = forward_backward(obs, hmm)

    assert posteriors.shape == (60, 4)
    np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(60), atol=1e-5)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    """save_hmm + load_hmm round-trips all arrays losslessly."""
    hmm = _make_hmm_state()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        save_hmm(hmm, path)
        loaded = load_hmm(path)

        assert loaded is not None
        assert loaded.n_states == hmm.n_states
        np.testing.assert_array_equal(loaded.means, hmm.means)
        np.testing.assert_array_equal(loaded.covariances, hmm.covariances)
        np.testing.assert_array_equal(loaded.transition, hmm.transition)
        np.testing.assert_array_equal(loaded.initial, hmm.initial)
    finally:
        path.unlink(missing_ok=True)


def test_load_hmm_missing_file_returns_none():
    """load_hmm returns None when file doesn't exist."""
    result = load_hmm(Path("/tmp/nonexistent_hmm_state_xyz.json"))
    assert result is None


def test_load_hmm_corrupt_file_returns_none():
    """load_hmm returns None on invalid JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{invalid json")
        path = Path(f.name)

    try:
        result = load_hmm(path)
        assert result is None
    finally:
        path.unlink(missing_ok=True)


def test_save_creates_parent_dirs():
    """save_hmm creates parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "hmm.json"
        hmm = _make_hmm_state()
        save_hmm(hmm, path)
        assert path.exists()
