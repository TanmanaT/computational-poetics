import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np
import time
from sklearn.utils.class_weight import compute_class_weight

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── NEURAL MODELS ────────────────────────────────────────────────────────────

class MLPModel(nn.Module):
    """
    Multilayer Perceptron (MLP) with Batch Normalization and Dropout.
    
    Designed for high-dimensional feature spaces (TF-IDF + Stylometrics).
    Uses BatchNorm1d to stabilize training against varying feature scales.
    
    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of target classes or 1 for regression.
        hidden_layers (list): Neurons per hidden layer.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        if x.size(0) > 1: return self.net(x)
        # Avoid BatchNorm failure on single-sample eval
        self.eval()
        with torch.no_grad(): return self.net(x)

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for Sequential Poetic Analysis.
    
    Treats the feature vector as a pseudo-sequence to capture rhythmic 
    and structural dependencies across a poem's latent space.
    
    Args:
        input_dim (int): Total dimensionality of the feature space.
        hidden_dim (int): LSTM hidden state dimension.
        num_layers (int): Number of recurrent layers.
        output_dim (int): Target projection dimension.
        seq_len (int): Number of virtual 'time-steps' to split the input into.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=6, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.step_dim = (input_dim + seq_len - 1) // seq_len
        self.padded_dim = self.step_dim * seq_len
        
        self.lstm = nn.LSTM(self.step_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        if x.size(1) < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        x_seq = x.view(batch_size, self.seq_len, self.step_dim)
        out, _ = self.lstm(x_seq)
        return self.fc(out[:, -1, :])

class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network for Structural Motif Detection.
    
    Utilizes localized kernels to detect n-gram like patterns in the 
    feature representation (stylometrics + character sequences).
    
    Args:
        input_dim (int): Total dimensionality of the input.
        output_dim (int): Target classes.
        seq_len (int): Virtual sequence mapping for 1D convolution.
        num_filters (int): Number of feature maps per layer.
    """
    def __init__(self, input_dim, output_dim, seq_len=8, num_filters=256):
        super().__init__()
        self.seq_len = seq_len
        self.step_dim = (input_dim + seq_len - 1) // seq_len
        self.padded_dim = self.step_dim * seq_len
        
        self.conv1 = nn.Conv1d(self.step_dim, num_filters, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(num_filters)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        if x.size(1) < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
            
        x_seq = x.view(batch_size, self.seq_len, self.step_dim).permute(0, 2, 1)
        
        x = torch.relu(self.bn1(self.conv1(x_seq)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ─── NEURAL WRAPPERS ──────────────────────────────────────────────────────────

class BaseTorchWrapper(BaseEstimator):
    """
    Scikit-learn compatible wrapper for PyTorch models.
    
    Implements the BaseEstimator interface to allow integration with 
    VotingClassifier, Pipeline, and GridSearchCV. 
    Includes built-in:
    - Early Stopping (Val Loss based)
    - Dynamic Class Balancing (Weighted Cross-Entropy)
    - Device Management (CUDA/CPU)
    """
    def __init__(self, model_class=None, input_dim=None, output_dim=None, is_regressor=False, **kwargs):
        self.model_class = model_class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_regressor = is_regressor
        self.kwargs = kwargs
        self.model = None
        self.classes_ = None

    def _more_tags(self):
        return {"estimator_type": "regressor" if self.is_regressor else "classifier"}

    @property
    def _estimator_type(self):
        return "regressor" if self.is_regressor else "classifier"

    def fit(self, X, y):
        if self.model is None:
             self.model = self.model_class(self.input_dim, self.output_dim, **self.kwargs).to(DEVICE)
        
        start_t = time.time()
        v_size = max(1, int(0.1 * len(X)))
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        val_idx, train_idx = indices[:v_size], indices[v_size:]
        
        X_tr_data = torch.tensor(X[train_idx], dtype=torch.float32).to(DEVICE)
        y_tr_data = torch.tensor(y[train_idx], dtype=torch.float32 if self.is_regressor else torch.long).to(DEVICE)
        X_val_data = torch.tensor(X[val_idx], dtype=torch.float32).to(DEVICE)
        y_val_data = torch.tensor(y[val_idx], dtype=torch.float32 if self.is_regressor else torch.long).to(DEVICE)

        dataset = TensorDataset(X_tr_data, y_tr_data)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        if not self.is_regressor:
            # Handle Imbalance: Compute class weights
            labels = y.astype(int) if hasattr(y, 'astype') else y
            u_classes = np.unique(labels)
            # Ensure indices match classes for weighted loss
            cw = compute_class_weight('balanced', classes=u_classes, y=labels)
            weights_t = torch.zeros(self.output_dim).to(DEVICE)
            for i, cls_idx in enumerate(u_classes):
                if cls_idx < self.output_dim: weights_t[cls_idx] = cw[i]
            criterion = nn.CrossEntropyLoss(weight=weights_t)
        else:
            criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 7
        no_improve = 0
        epochs = 100 
        
        print(f"\t[Neural Training] Input: {X_tr_data.shape[1]} | Target: {self.output_dim} | Device: {DEVICE}")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            self.model.train()
            running_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx).squeeze()
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                v_out = self.model(X_val_data).squeeze()
                val_loss = criterion(v_out, y_val_data).item()
            
            elapsed = time.time() - epoch_start
            if epoch % 5 == 0 or epoch == 1:
                print(f"\t\t[Epoch {epoch:3d}] Loss: {running_loss/len(loader):.5f} | Val: {val_loss:.5f} | {elapsed:.2f}s", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
            
            if no_improve >= patience:
                total_t = time.time() - start_t
                print(f"\tEarly stopping at {epoch}. Restored best state. (Total: {total_t:.1f}s)", flush=True)
                self.model.load_state_dict({k: v.to(DEVICE) for k, v in self.best_state.items()})
                break
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            out = self.model(X_t)
            if self.is_regressor: return out.cpu().numpy().flatten()
            else: return torch.argmax(out, dim=1).cpu().numpy()

    def predict_proba(self, X):
        if self.is_regressor: return None
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            # Add tiny epsilon to avoid potential division by zero in softmax
            logits = self.model(X_t)
            out = torch.softmax(logits, dim=1)
            # Ensure no NaNs propagate to Scikit-learn
            res = out.cpu().numpy()
            return np.nan_to_num(res, nan=0.0)

    def get_params(self, deep=True):
        return {"model_class": self.model_class, "input_dim": self.input_dim, "output_dim": self.output_dim, "is_regressor": self.is_regressor}
    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

class TorchClassifierWrapper(ClassifierMixin, BaseTorchWrapper):
    pass

class TorchRegressorWrapper(RegressorMixin, BaseTorchWrapper):
    pass

# Alias
TorchWrapper = TorchClassifierWrapper

# ─── LINEAR WRAPPERS ──────────────────────────────────────────────────────────

class BaseVerboseLinearWrapper(BaseEstimator):
    """
    Verbose Wrapper for Scikit-learn Linear Models (SGDClassifier/Regressor).
    
    Enables epoch-by-epoch logging of the training process for non-neural 
    heads using partial_fit. Provides transparency for ensemble weights evolution.
    """
    def __init__(self, estimator_class=None, is_regressor=False, **kwargs):
        self.estimator_class = estimator_class
        self.is_regressor = is_regressor
        self.kwargs = kwargs
        self.model = None

    def _more_tags(self):
        return {"estimator_type": "regressor" if self.is_regressor else "classifier"}

    @property
    def _estimator_type(self):
        return "regressor" if self.is_regressor else "classifier"

    def fit(self, X, y):
        if self.model is None:
            self.model = self.estimator_class(**self.kwargs)
        classes = np.unique(y) if not self.is_regressor else None
        epochs = 20
        name = self.estimator_class.__name__ if self.estimator_class else 'Model'
        print(f"\t  [Linear Training] Multi-step fit for {name}...")
        for epoch in range(1, epochs + 1):
            if self.is_regressor:
                self.model.partial_fit(X, y)
            else:
                self.model.partial_fit(X, y, classes=classes)
            if epoch % 5 == 0 or epoch == 1:
                print(f"\t\t[Step {epoch:2d}/20] Weights evolving...", flush=True)
        return self

    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
    def get_params(self, deep=True): return {"estimator_class": self.estimator_class, "is_regressor": self.is_regressor, **self.kwargs}
    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

class VerboseLinearClassifier(ClassifierMixin, BaseVerboseLinearWrapper):
    pass

class VerboseLinearRegressor(RegressorMixin, BaseVerboseLinearWrapper):
    pass

# Alias
VerboseLinearWrapper = VerboseLinearClassifier
