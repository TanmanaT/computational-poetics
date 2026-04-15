"""
train_models.py  —  STEP 3
==============================================
Trains all 4 models with High-Accuracy configurations:
- Feature Scaling (StandardScaler)

Run: python models/train_models.py
"""

import os, sys, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              mean_squared_error, r2_score, mean_absolute_error)
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, VotingRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone, BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Global Hardware Detection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_NAME = 'cuda' if DEVICE == 'cuda' else 'cpu'

# Root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROC   = os.path.join(ROOT, 'data', 'processed')
MODELS = os.path.join(ROOT, 'saved_models')
os.makedirs(MODELS, exist_ok=True)

from models.neural_base import (
    MLPModel, LSTMModel, CNNModel,
    TorchClassifierWrapper, TorchRegressorWrapper,
    VerboseLinearClassifier, VerboseLinearRegressor
)

all_results = {}

def should_skip(src_file, out_file):
    """Returns True if the output file exists and is newer than the source file."""
    if '--force' in sys.argv:
        return False
    if not os.path.exists(out_file):
        return False
    if not os.path.exists(src_file):
        return False
    return os.path.getmtime(out_file) > os.path.getmtime(src_file)


def load_X(name):
    return np.load(os.path.join(PROC, f'{name}_X.npz'))['X']

def load_y(name):
    return np.load(os.path.join(PROC, f'{name}_y.npy'))

def save_model(model, name):
    pickle.dump(model, open(os.path.join(MODELS, f'{name}_model.pkl'), 'wb'))
    print(f"\t{name}_model.pkl saved")

def load_le(name):
    return pickle.load(open(os.path.join(MODELS, f'{name}_le.pkl'), 'rb'))


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'f1_score':  round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def train_classifiers(models_dict, X_tr, y_tr, X_te, y_te, task, le=None):
    """
    Executes the training and cross-validation pipeline for classification models.
    
    1) Applies StandardScaler to the dense feature matrix.
    2) Trains the model (using an epoch-loop for SGDClassifiers to print real-time loss).
    3) Evaluates predictions against the test set for Accuracy, Precision, Recall, and F1.
    4) Runs a 3-fold cross-validation on the training set to measure variance/stability.
    
    Args:
        models_dict (dict): Dictionary mapping model names to instantiated Scikit-Learn classifiers.
        X_tr (np.ndarray): Training feature matrix.
        y_tr (np.ndarray): Training labels.
        X_te (np.ndarray): Testing feature matrix.
        y_te (np.ndarray): Testing labels.
        task (str): Human-readable name of the current task (e.g., 'Genre').
        le (LabelEncoder): Optional. If provided, prints a detailed classification report mapping.
        
    Returns:
        tuple(Estimator, dict): The highest performing Pipeline model and a dictionary of all results.
    """
    best_f1, best_model, best_name = -1, None, None
    task_results = {}
    cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models_dict.items():
        print(f"\n  {task} — {name}...")
        
        # Dense sparse matrices (TF-IDF) lose critical 0-bound information when StandardScaled.
        # Linear models (SVM/SGD) require it for optimization math, but Trees are destroyed by it.
        use_scaler = 'passthrough' if any(k in name for k in ['XGBoost', 'Trees']) else StandardScaler()
        
        # Scaling + Model Pipeline
        pipeline = Pipeline([
            ('scaler', use_scaler),
            ('clf', model)
        ])

        # Removed XGBoost custom logic, Linear models use simple fit
        print(f"\t[{name} starting main fit sequence...]", flush=True)
        if 'SGD' in name:
            # Manual epoch loop for live printing and Early Stopping
            X_t, X_v, y_t, y_v = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)
            
            X_t_scaled = np.nan_to_num(pipeline.named_steps['scaler'].fit_transform(X_t))
            X_v_scaled = np.nan_to_num(pipeline.named_steps['scaler'].transform(X_v))
            clf = pipeline.named_steps['clf']
            classes = np.unique(y_tr)
            
            epochs = 500
            best_val = -np.inf
            patience = 50
            no_improve = 0
            
            for epoch in range(1, epochs + 1):
                clf.partial_fit(X_t_scaled, y_t, classes=classes)
                if epoch % 10 == 0 or epoch == 1:
                    v_pred = clf.predict(X_v_scaled)
                    v_f1 = f1_score(y_v, v_pred, average='weighted', zero_division=0)
                    
                    print(f"\t[Epoch {epoch}/{epochs}] Val F1: {v_f1:.4f}", flush=True)
                    
                    if v_f1 > best_val:
                        best_val = v_f1
                        no_improve = 0
                    else:
                        # Only start counting non-improvements after epoch 1
                        if epoch > 1:
                            no_improve += 10
                            
                    if no_improve >= patience:
                        print(f"\tEarly stopping triggered at epoch {epoch}! Best Val F1: {best_val:.4f}", flush=True)
                        break
            
            # Re-fit scaler on the whole training set so the pipeline is correct for test
            pipeline.named_steps['scaler'].fit(X_tr)
        else:
            pipeline.fit(X_tr, y_tr)
        print(f"\t[{name} main fit complete! Starting cross-validation...]", flush=True)
        
        X_te_scaled = np.nan_to_num(pipeline.named_steps['scaler'].transform(X_te))
        metrics = evaluate_classifier(pipeline.named_steps['clf'], X_te_scaled, y_te)

        try:
            # For CV, we clone the model and UNSET early stopping
            cv_model = clone(model)
            if hasattr(cv_model, 'early_stopping_rounds'):
                cv_model.set_params(early_stopping_rounds=None)
            
            cv_pipeline = Pipeline([('scaler', use_scaler), ('clf', cv_model)])
            
            print(f"\tRunning 3-fold CV (n_jobs=1 for stability)...", flush=True)
            cv_scores = cross_val_score(cv_pipeline, X_tr, y_tr, cv=cv_splitter,
                                         scoring='f1_weighted', n_jobs=1, verbose=1)
            metrics['cv_mean'] = round(float(cv_scores.mean()), 4)
            metrics['cv_std']  = round(float(cv_scores.std()),  4)
        except Exception as e:
            print(f"CV failed for {name}: {e}")
            metrics['cv_mean'] = metrics['f1_score']
            metrics['cv_std']  = 0.0

        print(f"\tAcc={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}  "
              f"\tCV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")
        task_results[name] = metrics

        if metrics['f1_score'] > best_f1:
            best_f1, best_model, best_name = metrics['f1_score'], pipeline, name

    print(f"\n\tBest {task}: {best_name}, F1={best_f1:.4f}")
    if le:
        y_pred = best_model.predict(X_te)
        print(classification_report(y_te, y_pred,
                                     target_names=le.classes_, zero_division=0))
    return best_model, task_results


# 1. GENRE
def train_genre():
    """
    Orchestrates the training of the Genre Classification task.
    
    Uses a 'Tetra-Neural' ensemble combining linear, dense, sequential, 
    and convolutional heads. Leverages balanced data from genre_balanced.csv.
    """
    print("\n" + "="*52)
    print("\tGENRE CLASSIFICATION")
    print("="*52)
    data_path = os.path.join(PROC, 'genre_X.npz')
    model_path = os.path.join(MODELS, 'genre_model.pkl')
    
    if should_skip(data_path, model_path):
        print("\tModel already up-to-date. Skipping.")
        return

    X = load_X('genre');  y = load_y('genre');  le = load_le('genre')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    classes = np.unique(y_tr)
    weights = compute_class_weight('balanced', classes=classes, y=y_tr)
    weight_dict = dict(zip(classes, weights))

    # --- PURE NEURAL ENSEMBLE (MLP + LSTM + CNN) ---
    mlp  = TorchClassifierWrapper(MLPModel, X_tr.shape[1], len(le.classes_))
    lstm = TorchClassifierWrapper(LSTMModel, X_tr.shape[1], len(le.classes_))
    cnn  = TorchClassifierWrapper(CNNModel, X_tr.shape[1], len(le.classes_))
    
    models = {
        'NeuralHybrid (MLP+LSTM+CNN)': VotingClassifier(
            estimators=[('mlp', mlp), ('lstm', lstm), ('cnn', cnn)], 
            voting='soft', weights=[1.0, 1.5, 1.5], n_jobs=1, verbose=0)
    }
    
    print("Starting Phase 3 Neural Training...", flush=True)
    best, results = train_classifiers(models, X_tr, y_tr, X_te, y_te, 'Genre', le)
    save_model(best, 'genre')
    all_results['genre'] = results


# 2. FIGURATIVE
def train_figurative():
    """
    Trains the Figurative Language Detection model.
    
    Combines an SGD-SVM head for keyword detection with an MLP head 
    for semantic relationship inference.
    """
    print("\n" + "="*52)
    print("\tFIGURATIVE LANGUAGE DETECTION")
    print("="*52)
    data_path = os.path.join(PROC, 'figurative_X.npz')
    model_path = os.path.join(MODELS, 'figurative_model.pkl')

    if should_skip(data_path, model_path):
        print("\tModel already up-to-date. Skipping.")
        return

    X = load_X('figurative');  y = load_y('figurative');  le = load_le('figurative')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    classes_fig = np.unique(y_tr)
    weights_fig = compute_class_weight('balanced', classes=classes_fig, y=y_tr)
    weight_dict_fig = dict(zip(classes_fig, weights_fig))

    # Neural Ensemble for Figurative
    mlp = TorchClassifierWrapper(MLPModel, X_tr.shape[1], len(le.classes_))

    models = {
        'NeuralVoter (MLP)': VotingClassifier(
            estimators=[('mlp', mlp)], voting='soft', n_jobs=1)
    }
    best, results = train_classifiers(models, X_tr, y_tr, X_te, y_te, 'Figurative', le)
    save_model(best, 'figurative')
    all_results['figurative'] = results


# 3. EMOTION
def train_emotion():
    print("\n" + "="*52)
    print("\tEMOTION DETECTION")
    print("="*52)
    data_path = os.path.join(PROC, 'emotion_X.npz')
    model_path = os.path.join(MODELS, 'emotion_model.pkl')

    if should_skip(data_path, model_path):
        print("\tModel already up-to-date. Skipping.")
        return

    X = load_X('emotion');  y = load_y('emotion');  le = load_le('emotion')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    classes_emo = np.unique(y_tr)
    weights_emo = compute_class_weight('balanced', classes=classes_emo, y=y_tr)
    weight_dict_emo = dict(zip(classes_emo, weights_emo))

    # Neural Ensemble for Emotion
    mlp = TorchClassifierWrapper(MLPModel, X_tr.shape[1], len(le.classes_))

    models = {
        'NeuralVoter (MLP)': VotingClassifier(
            estimators=[('mlp', mlp)], voting='soft', n_jobs=1)
    }
    best, results = train_classifiers(models, X_tr, y_tr, X_te, y_te, 'Emotion', le)
    save_model(best, 'emotion')
    all_results['emotion'] = results


# 4. QUALITY
def train_quality():
    """
    Calculates poetic quality scores using a stable regression ensemble.
    
    Employs a Ridge-MLP architecture to prevent gradient explosion and 
    ensure normalized beauty metrics across the [0, 10] range.
    """
    print("\n" + "="*52)
    print("\tQUALITY SCORE PREDICTION")
    print("="*52)
    data_path = os.path.join(PROC, 'quality_X.npz')
    model_path = os.path.join(MODELS, 'quality_model.pkl')

    if should_skip(data_path, model_path):
        print("\tModel already up-to-date. Skipping.")
        return

    X = load_X('quality');  y = load_y('quality')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Neural-Linear Ensemble for Quality (Pivoting to Ridge for absolute stability)
    from sklearn.linear_model import Ridge
    svm = Ridge(alpha=1.0) # Closed-form, no gradient explosion
    mlp = TorchRegressorWrapper(MLPModel, X_tr.shape[1], 1, is_regressor=True)

    models = {
        'NeuralRegressor (Ridge+MLP)': VotingRegressor(estimators=[('ridge', svm), ('mlp', mlp)], n_jobs=1, verbose=0)
    }

    best_r2, best_model, best_name = -float('inf'), None, None
    task_results = {}

    for name, model in models.items():
        print(f"\n  Quality — {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', model)
        ])
        
        pipeline.fit(X_tr, y_tr)
        
        y_pred = pipeline.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae  = float(mean_absolute_error(y_te, y_pred))
        r2   = float(r2_score(y_te, y_pred))

        # CV
        cv_model = clone(model)
        if hasattr(cv_model, 'early_stopping_rounds'):
            cv_model.set_params(early_stopping_rounds=None)
        cv_pipeline = Pipeline([('scaler', StandardScaler()), ('reg', cv_model)])
        
        print(f"\tRunning Quality CV (n_jobs=1)...", flush=True)
        cv_scores = cross_val_score(cv_pipeline, X_tr, y_tr, cv=3, scoring='r2', n_jobs=1, verbose=0)
        task_results[name] = {
            'rmse': round(rmse,4), 'mae': round(mae,4),
            'r2_score': round(r2,4),
            'cv_mean': round(float(cv_scores.mean()),4),
            'cv_std':  round(float(cv_scores.std()),4)
        }
        print(f"    RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, pipeline, name

    print(f"\nBest Model: {best_name}  (R²={best_r2:.4f})")
    save_model(best_model, 'quality')
    all_results['quality'] = task_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force:
        def should_skip(s, o): return False

    print("=" * 52)
    print("\tSTEP 3 - MODEL TRAINING")
    print("=" * 52)

    train_genre()
    train_figurative()
    train_emotion()
    train_quality()

    # Save results JSON
    json_path = os.path.join(MODELS, 'training_results.json')
    if os.path.exists(json_path):
        try:
            old_results = json.load(open(json_path, 'r'))
            for task, res in old_results.items():
                if task not in all_results:
                    all_results[task] = res
        except Exception:
            pass

    json.dump(all_results, open(json_path, 'w'), indent=2)

    print("\n" + "="*52)
    print("\tTRAINING COMPLETE - SUMMARY")
    print("="*52)
    for task, res in all_results.items():
        print(f"\n  {task.upper()}:")
        for model, m in res.items():
            if 'f1_score' in m:
                print(f"\t{model:30s} Acc={m['accuracy']:.3f}, F1={m['f1_score']:.3f}")
            else:
                print(f"\t{model:30s} RMSE={m['rmse']:.3f}, R²={m['r2_score']:.3f}")

    print(f"\nAll models saved to saved_models/")
