import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple, Dict, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix as cm, roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize

class DataPreprocessor:
    """Data preprocessing pipeline for ML models"""
    
    def __init__(self):
        self.scaler_mean = None
        self.scaler_std = None
        self.label_encoders = {}
        self.fitted = False
    
    def encode_categorical(self, X: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical columns to numeric values
        
        Args:
            X: Input DataFrame
            categorical_cols: List of categorical column names
            
        Returns:
            pandas.DataFrame: Encoded DataFrame
        """
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                # Fit encoder
                unique_vals = X[col].unique()
                self.label_encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
            
            # Transform
            X_encoded[col] = X[col].map(self.label_encoders[col])
        
        return X_encoded
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features to mean=0, std=1
        
        Args:
            X: Input feature array
            fit: Whether to fit the scaler or use existing parameters
            
        Returns:
            numpy.ndarray: Scaled features
        """
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-9
            return (X - self.scaler_mean) / self.scaler_std
        else:
            return (X - self.scaler_mean) / self.scaler_std
    
    def preprocess(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        categorical_cols: List[str],
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline
        
        Args:
            X: Input features
            y: Target labels
            categorical_cols: List of categorical column names
            scale: Whether to scale features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed features and labels
        """
        # Encode categorical
        X = self.encode_categorical(X, categorical_cols)
        
        # Convert to numpy
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = X.astype(float)
        
        # Scale features
        if scale:
            X = self.scale_features(X, fit=True)
        
        self.fitted = True
        return X, y
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray], categorical_cols: List[str]) -> np.ndarray:
        """
        Apply learned preprocessing to new data
        
        Args:
            X: Input features to transform
            categorical_cols: List of categorical column names
            
        Returns:
            numpy.ndarray: Transformed features
            
        Raises:
            ValueError: If preprocessor is not fitted
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call preprocess first.")
        
        X = self.encode_categorical(X, categorical_cols)
        
        if hasattr(X, 'values'):
            X = X.values
        
        X = X.astype(float)
        X = self.scale_features(X, fit=False)
        
        return X


def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets
    
    Args:
        X: Input features
        y: Target labels
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class SMOTE:
    """Synthetic Minority Over-sampling Technique for handling class imbalance"""
    
    def __init__(self, k_neighbors: int = 5):
        """
        Initialize SMOTE
        
        Args:
            k_neighbors: Number of nearest neighbors for synthetic generation
        """
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance classes
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Resampled features and labels
        """
        X = np.array(X)
        y = np.array(y)
        
        classes, counts = np.unique(y, return_counts=True)
        
        if len(classes) < 2:
            return X, y
        
        max_count = max(counts)
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        for class_label, count in zip(classes, counts):
            if count < max_count:
                # Find minority class samples
                minority_idx = np.where(y == class_label)[0]
                minority_X = X[minority_idx]
                
                # Generate synthetic samples
                n_synthetic = max_count - count
                synthetic_X = self._generate_synthetic(minority_X, n_synthetic)
                synthetic_y = np.array([class_label] * n_synthetic)
                
                X_resampled = np.vstack([X_resampled, synthetic_X])
                y_resampled = np.concatenate([y_resampled, synthetic_y])
        
        return X_resampled, y_resampled
    
    def _generate_synthetic(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate synthetic samples using nearest neighbors
        
        Args:
            X: Minority class samples
            n_samples: Number of synthetic samples to generate
            
        Returns:
            numpy.ndarray: Synthetic samples
        """
        synthetic = []
        
        for _ in range(n_samples):
            # Random existing sample
            idx = np.random.randint(len(X))
            x = X[idx]
            
            # Find nearest neighbor
            distances = np.sqrt(np.sum((X - x) ** 2, axis=1))
            sorted_indices = np.argsort(distances)
            neighbor_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            neighbor = X[neighbor_idx]
            
            # Create synthetic sample (interpolation)
            alpha = np.random.random()
            synthetic_sample = x + alpha * (neighbor - x)
            synthetic.append(synthetic_sample)
        
        return np.array(synthetic)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict: Dictionary containing accuracy, precision, recall, f1
    """
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Generate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    return cm(y_true, y_pred)


def calculate_roc_auc(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    n_classes: Optional[int] = None
) -> Dict[str, Union[dict, float]]:
    """
    Calculate ROC curve and AUC score for multi-class classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (shape: n_samples x n_classes)
        n_classes: Number of classes (auto-detected if None)
        
    Returns:
        Dict: ROC curves, AUC scores for each class, macro AUC, weighted AUC
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Determine number of classes
    if n_classes is None:
        n_classes = len(np.unique(y_true))
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
    
    # Calculate macro and weighted AUC
    macro_auc = np.mean(list(roc_auc.values()))
    
    # Weighted AUC (by class support)
    class_counts = np.bincount(y_true)
    weights = class_counts / len(y_true)
    weighted_auc = np.sum([roc_auc[i] * weights[i] for i in range(n_classes)])
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc_per_class': roc_auc,
        'macro_auc': round(macro_auc, 4),
        'weighted_auc': round(weighted_auc, 4),
        'n_classes': n_classes
    }


def calculate_multiclass_roc_auc_ovr(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Union[dict, float]]:
    """
    Calculate One-vs-Rest (OvR) AUC for multi-class classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dict: OvR AUC per class and macro OvR AUC
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n_classes = y_pred_proba.shape[1]
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate OvR AUC
    ovr_auc = {}
    for i in range(n_classes):
        ovr_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
    
    return {
        'ovr_auc_per_class': ovr_auc,
        'macro_ovr_auc': round(np.mean(list(ovr_auc.values())), 4)
    }


def calculate_auc_ci(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    confidence: float = 0.95
) -> Dict[str, Union[float, tuple]]:
    """
    Calculate AUC with confidence interval using bootstrapping
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Dict: Macro AUC mean, std, and confidence interval
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n_classes = y_pred_proba.shape[1]
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    n_bootstraps = 100
    macro_auc_scores = []
    
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        y_true_boot = y_true_bin[indices]
        y_pred_boot = y_pred_proba[indices]
        
        try:
            aucs = []
            for i in range(n_classes):
                auc = roc_auc_score(y_true_boot[:, i], y_pred_boot[:, i])
                aucs.append(auc)
            macro_auc_scores.append(np.mean(aucs))
        except:
            continue
    
    macro_auc_scores = np.array(macro_auc_scores)
    alpha = 1 - confidence
    lower_bound = np.percentile(macro_auc_scores, 100 * alpha / 2)
    upper_bound = np.percentile(macro_auc_scores, 100 * (1 - alpha / 2))
    
    return {
        'macro_auc_mean': round(np.mean(macro_auc_scores), 4),
        'macro_auc_std': round(np.std(macro_auc_scores), 4),
        'confidence_interval': (round(lower_bound, 4), round(upper_bound, 4)),
        'confidence_level': confidence
    }


def calculate_complete_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_pred_proba: Optional[np.ndarray] = None, 
    n_classes: Optional[int] = None
) -> Dict[str, Union[float, dict]]:
    """
    Calculate complete set of metrics including multi-class AUC
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (required for AUC)
        n_classes: Number of classes (auto-detected if None)
        
    Returns:
        Dict: Complete metrics including accuracy, precision, recall, f1, and AUC
    """
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    }
    
    # Add multi-class AUC metrics if probabilities are provided
    if y_pred_proba is not None:
        n_classes = n_classes or len(np.unique(y_true))
        auc_result = calculate_roc_auc(y_true, y_pred_proba, n_classes)
        metrics['macro_auc'] = auc_result['macro_auc']
        metrics['weighted_auc'] = auc_result['weighted_auc']
        metrics['auc_per_class'] = auc_result['auc_per_class']
    
    return metrics


def calculate_bias_variance(
    y_true: np.ndarray, 
    y_pred_train: np.ndarray, 
    y_pred_val: np.ndarray
) -> Dict[str, Union[float, str]]:
    """
    Calculate bias and variance for model evaluation
    
    Args:
        y_true: True labels
        y_pred_train: Predictions on training set
        y_pred_val: Predictions on validation set
        
    Returns:
        Dict: Bias, variance, and diagnosis with suggestion
    """
    from sklearn.metrics import accuracy_score
    
    train_accuracy = accuracy_score(y_true[:len(y_pred_train)], y_pred_train)
    val_accuracy = accuracy_score(y_true[len(y_pred_train):len(y_pred_train)+len(y_pred_val)], y_pred_val)
    
    bias = 1 - train_accuracy
    variance = train_accuracy - val_accuracy
    
    # Diagnose overfitting/underfitting
    if variance > 0.1 and train_accuracy > 0.9:
        status = "Overfitting"
        suggestion = "Increase regularization, reduce model complexity, or add more data"
    elif bias > 0.3:
        status = "Underfitting"
        suggestion = "Increase model complexity, add more features, or reduce regularization"
    else:
        status = "Balanced"
        suggestion = "Model is well-tuned"
    
    return {
        'train_accuracy': round(train_accuracy, 4),
        'val_accuracy': round(val_accuracy, 4),
        'bias': round(bias, 4),
        'variance': round(variance, 4),
        'status': status,
        'suggestion': suggestion
    }


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa for inter-rater agreement
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Kappa score (0-1)
    """
    from sklearn.metrics import cohen_kappa_score as kappa
    return round(kappa(y_true, y_pred), 4)


def calculate_micro_macro_averages(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate micro and macro averages for precision, recall, f1
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict: Micro and macro averages for precision, recall, f1
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'micro': {
            'precision': round(precision_micro, 4),
            'recall': round(recall_micro, 4),
            'f1': round(f1_micro, 4)
        },
        'macro': {
            'precision': round(precision_macro, 4),
            'recall': round(recall_macro, 4),
            'f1': round(f1_macro, 4)
        }
    }


def convert_to_2d_confusion_matrix(
    confusion_matrix: np.ndarray, 
    max_classes: int = 10
) -> np.ndarray:
    """
    Convert high-dimensional confusion matrix to 2D for visualization
    
    Args:
        confusion_matrix: numpy array of shape (n_classes, n_classes)
        max_classes: Maximum classes to show (if more, aggregate)
        
    Returns:
        numpy.ndarray: 2D array suitable for visualization
    """
    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]
    
    if n_classes <= max_classes:
        return cm
    
    # Aggregate smaller classes
    n_groups = max_classes
    group_size = n_classes // n_groups
    
    reduced_cm = np.zeros((n_groups, n_groups))
    
    for i in range(n_groups):
        for j in range(n_groups):
            i_start = i * group_size
            i_end = min((i + 1) * group_size, n_classes)
            j_start = j * group_size
            j_end = min((j + 1) * group_size, n_classes)
            reduced_cm[i, j] = np.sum(cm[i_start:i_end, j_start:j_end])
    
    return reduced_cm