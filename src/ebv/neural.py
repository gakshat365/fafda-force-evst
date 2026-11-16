"""A small multilayer perceptron in PyTorch with a scikit-learn interface.

It behaves like any scikit-learn regressor (fit / predict / get_params), so it plugs
into the same tuning and cross-validation code as every other model. Inputs are
expected to be standardised already (it is used after a StandardScaler); the target is
standardised internally. Training uses AdamW with weight decay and early stopping on a
held-back validation share, then restores the best weights.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden_layers=(64, 32), weight_decay=1e-3, learning_rate=1e-3, dropout=0.0,
                 max_epochs=1500, patience=60, validation_fraction=0.15, random_state=42):
        self.hidden_layers = hidden_layers
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _network(self, n_inputs: int) -> nn.Sequential:
        layers, width = [], n_inputs
        for size in self.hidden_layers:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        return nn.Sequential(*layers, nn.Linear(width, 1))

    def fit(self, X, y):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        y_scaled = (y - self.y_mean_) / self.y_std_

        order = rng.permutation(len(X))
        n_valid = max(1, int(len(X) * self.validation_fraction))
        valid, train = order[:n_valid], order[n_valid:]
        to_tensor = lambda a: torch.as_tensor(a, device=DEVICE)
        X_train, y_train = to_tensor(X[train]), to_tensor(y_scaled[train]).unsqueeze(1)
        X_valid, y_valid = to_tensor(X[valid]), to_tensor(y_scaled[valid]).unsqueeze(1)

        self.network_ = self._network(X.shape[1]).to(DEVICE)
        optimiser = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best_loss, best_state, waited = np.inf, None, 0
        for _ in range(self.max_epochs):               # full-batch training: the data set is small
            self.network_.train()
            optimiser.zero_grad()
            loss_fn(self.network_(X_train), y_train).backward()
            optimiser.step()
            self.network_.eval()
            with torch.no_grad():
                valid_loss = loss_fn(self.network_(X_valid), y_valid).item()
            if valid_loss < best_loss - 1e-5:
                best_loss, waited = valid_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.network_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.network_.load_state_dict(best_state)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        self.network_.eval()
        with torch.no_grad():
            out = self.network_(torch.as_tensor(np.asarray(X, np.float32), device=DEVICE))
        return out.squeeze(1).cpu().numpy() * self.y_std_ + self.y_mean_
