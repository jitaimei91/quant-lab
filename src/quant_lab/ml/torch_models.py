"""PyTorch model architectures wrapped in a sklearn-compatible fit/predict API.

Each architecture class subclasses `_TorchPredictor` and implements
`_build_module(input_dim) -> nn.Module`. The base class handles training
loop, batching, early stopping, prediction, and joblib persistence.

The lineage here is qlib's pytorch model zoo, but the qlib classes hard-
depend on its DatasetH/DataHandlerLP framework. Re-implementing the
architectures directly keeps the integration light: no qlib at runtime,
no binary data format, no DataHandlerLP. The walk-forward training,
gating, and ensemble plumbing that already exists for XGB/LGBM applies
unchanged.
"""
from __future__ import annotations

import numpy as np


_DEFAULT_EPOCHS = 30
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_LR = 1e-3
_DEFAULT_WEIGHT_DECAY = 1e-5
_DEFAULT_EARLY_STOP_PATIENCE = 5


class _TorchPredictor:
    """sklearn-style wrapper around an nn.Module.

    Subclasses override `_build_module(input_dim)` to return the architecture.
    The base class trains with MSE loss + Adam, holds out 10% for early
    stopping, and exposes `.predict(numpy_array)`.
    """

    epochs: int = _DEFAULT_EPOCHS
    batch_size: int = _DEFAULT_BATCH_SIZE
    lr: float = _DEFAULT_LR
    weight_decay: float = _DEFAULT_WEIGHT_DECAY
    patience: int = _DEFAULT_EARLY_STOP_PATIENCE

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.module = None
        self.input_dim = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def _build_module(self, input_dim: int):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchPredictor":
        import torch
        from torch import nn

        # Avoid OpenMP deadlock when torch coexists with libgomp-linked
        # libraries (lightgbm, catboost) in the same process. Common pain
        # on macOS where libomp and libgomp can collide.
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        np_rng = np.random.default_rng(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n, d = X.shape
        self.input_dim = d

        # Per-feature standardization — neural nets are sensitive to scale.
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        Xs = (X - self._x_mean) / self._x_std

        # 90/10 train/validation split (deterministic)
        idx = np_rng.permutation(n)
        n_val = max(1, n // 10)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xt = torch.from_numpy(Xs[tr_idx])
        yt = torch.from_numpy(y[tr_idx])
        Xv = torch.from_numpy(Xs[val_idx])
        yv = torch.from_numpy(y[val_idx])

        self.module = self._build_module(d)
        opt = torch.optim.Adam(
            self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        for epoch in range(self.epochs):
            self.module.train()
            perm = torch.randperm(len(Xt))
            for start in range(0, len(Xt), self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                xb, yb = Xt[batch_idx], yt[batch_idx]
                opt.zero_grad()
                pred = self.module(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.module.eval()
            with torch.no_grad():
                val_pred = self.module(Xv).squeeze(-1)
                val_loss = float(loss_fn(val_pred, yv))
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.module.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.module is None or self._x_mean is None:
            return np.zeros(len(X), dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)
        Xs = (X - self._x_mean) / self._x_std
        with torch.no_grad():
            out = self.module(torch.from_numpy(Xs)).squeeze(-1).cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class MLPPredictor(_TorchPredictor):
    """3-layer feedforward MLP with dropout. Tabular input only.

    Matches qlib's pytorch_nn (GeneralPTNN) in spirit: a small MLP with
    a configurable hidden size, dropout, and Adam optimizer.
    """

    hidden_dim: int = 64
    dropout: float = 0.2

    def _build_module(self, input_dim: int):
        from torch import nn

        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )


def fit_mlp(X: np.ndarray, y: np.ndarray, seed: int = 42) -> MLPPredictor:
    """Convenience: train an MLPPredictor and return it."""
    return MLPPredictor(seed=seed).fit(X, y)
