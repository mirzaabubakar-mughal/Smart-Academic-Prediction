import numpy as np
from collections import Counter
from math import sqrt
from typing import Union, List, Tuple, Optional


class KNN:
    """K-Nearest Neighbors with multiple distance metrics"""
    
    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        """
        Initialize KNN classifier
        
        Args:
            k: Number of nearest neighbors
            metric: Distance metric ('euclidean' or 'manhattan')
        """
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store training data (lazy learning)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance: sqrt(sum((a-b)^2))"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    def _manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Manhattan distance: sum(|a-b|)"""
        return np.sum(np.abs(a - b))
    
    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance based on selected metric"""
        if self.metric == 'manhattan':
            return self._manhattan_distance(a, b)
        else:
            return self._euclidean_distance(a, b)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using nearest neighbors
        
        Args:
            X: Input features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = [self._calculate_distance(x, train_x) for train_x in self.X_train]
            
            # Find k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            
            # Vote for most common label
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes for discrete features"""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes
        
        Args:
            alpha: Laplace smoothing parameter
        """
        self.alpha = alpha
        self.classes = None
        self.feature_probs = None
        self.priors = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate probability distributions for each class
        
        Args:
            X: Input features
            y: Target labels
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        self.feature_probs = []
        self.priors = []
        
        for c in self.classes:
            X_c = X[y == c]
            feature_count = np.sum(X_c, axis=0)
            total_count = np.sum(feature_count)
            prob = (feature_count + self.alpha) / (total_count + self.alpha * X.shape[1])
            self.feature_probs.append(prob)
            self.priors.append(len(X_c) / len(X))
        
        self.feature_probs = np.array(self.feature_probs)
        self.priors = np.array(self.priors)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using multinomial distribution
        
        Args:
            X: Input features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                log_likelihood = np.sum(x * np.log(self.feature_probs[i] + 1e-10))
                log_prior = np.log(self.priors[i])
                posterior = log_prior + log_likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)


class KernelSVM:
    """SVM with RBF Kernel for non-linear classification"""
    
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        epochs: int = 100, 
        lambda_param: float = 0.01, 
        gamma: float = 0.1
    ):
        """
        Initialize Kernel SVM
        
        Args:
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            lambda_param: Regularization parameter
            gamma: RBF kernel parameter
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.X_train = None
        self.y_train = None
        self.classes = None
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF (Gaussian) kernel"""
        diff = x1 - x2
        return np.exp(-self.gamma * np.dot(diff, diff))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train kernel SVM
        
        Args:
            X: Input features
            y: Target labels
        """
        X = np.array(X)
        y = np.array(y)
        self.X_train = X
        self.classes = np.unique(y)
        
        # For each class, train one-vs-rest
        self.models = {}
        for c in self.classes:
            y_binary = np.where(y == c, 1, -1)
            self._train_binary_kernel(X, y_binary, c)
    
    def _train_binary_kernel(self, X: np.ndarray, y: np.ndarray, class_label: Union[str, int]) -> None:
        """
        Train binary kernel SVM using SGD
        
        Args:
            X: Input features
            y: Binary labels (1 for class, -1 for others)
            class_label: The class being trained
        """
        n_samples = X.shape[0]
        alphas = np.zeros(n_samples)
        bias = 0
        
        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Compute decision function
                decision = bias
                for j in range(n_samples):
                    if alphas[j] != 0:
                        decision += alphas[j] * y[j] * self._rbf_kernel(X[i], X[j])
                
                # Update if margin violated
                if y[i] * decision < 1:
                    alphas[i] += self.lr * (1 - self.lambda_param * alphas[i])
                    bias += self.lr * y[i]
                else:
                    alphas[i] -= self.lr * self.lambda_param * alphas[i]
        
        self.models[class_label] = (alphas, bias)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using kernel SVM
        
        Args:
            X: Input features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            scores = {}
            for c, (alphas, bias) in self.models.items():
                score = bias
                for j, alpha in enumerate(alphas):
                    if alpha != 0:
                        score += alpha * self._rbf_kernel(x, self.X_train[j])
                scores[c] = score
            predictions.append(max(scores, key=scores.get))
        
        return np.array(predictions)


class SimpleNeuralNetwork:
    """Simple Neural Network with 1 hidden layer - Softmax only"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 16, 
        output_size: int = 5, 
        learning_rate: float = 0.01, 
        epochs: int = 100, 
        activation: str = 'relu'
    ):
        """
        Initialize Simple Neural Network
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden layer neurons
            output_size: Number of output classes
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            activation: Activation function for hidden layer
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.activation_name = activation
        
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.classes = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.loss_history = []
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _activate_hidden(self, x: np.ndarray) -> np.ndarray:
        """Apply hidden layer activation function (ReLU)"""
        return self._relu(x)
    
    def _activate_hidden_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of hidden layer activation function"""
        return self._relu_derivative(x)
    
    def _activate_output(self, x: np.ndarray) -> np.ndarray:
        """Output layer activation (always softmax for multi-class)"""
        return self._softmax(x)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train neural network using backpropagation
        
        Args:
            X: Input features
            y: Target labels
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to indices
        self.classes = np.unique(y)
        for i, c in enumerate(self.classes):
            self.class_to_idx[c] = i
            self.idx_to_class[i] = c
        
        y_encoded = np.array([self.class_to_idx[label] for label in y])
        
        # One-hot encode
        n_samples = len(X)
        n_classes = len(self.classes)
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y_encoded] = 1
        
        # Training loop
        for epoch in range(self.epochs):
            # Forward propagation
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._activate_hidden(z1)
            
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._activate_output(z2)
            
            # Calculate loss (cross-entropy)
            loss = -np.mean(np.sum(y_one_hot * np.log(a2 + 1e-8), axis=1))
            self.loss_history.append(loss)
            
            # Backpropagation
            m = n_samples
            
            # Output layer error
            dz2 = (a2 - y_one_hot) / m
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            # Hidden layer error
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self._activate_hidden_derivative(a1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights with gradient clipping
            dW2 = np.clip(dW2, -1, 1)
            dW1 = np.clip(dW1, -1, 1)
            
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained network
        
        Args:
            X: Input features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        X = np.array(X)
        # Forward pass
        a1 = self._activate_hidden(np.dot(X, self.W1) + self.b1)
        a2 = self._activate_output(np.dot(a1, self.W2) + self.b2)
        
        # Get class predictions
        predictions = np.argmax(a2, axis=1)
        return np.array([self.idx_to_class[i] for i in predictions])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        X = np.array(X)
        a1 = self._activate_hidden(np.dot(X, self.W1) + self.b1)
        a2 = self._activate_output(np.dot(a1, self.W2) + self.b2)
        return a2
    
    def get_loss_history(self) -> List[float]:
        """Return loss history for visualization"""
        return self.loss_history


class DecisionTreeClassifier:
    """Decision Tree for classification"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        """
        Initialize Decision Tree
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build decision tree
        
        Args:
            X: Input features
            y: Target labels
        """
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        """
        Recursively build tree
        
        Args:
            X: Input features
            y: Target labels
            depth: Current tree depth
            
        Returns:
            dict: Tree node
        """
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        
        # Stopping conditions
        if len(unique_classes) == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # Find best split
        best_feature, best_threshold, best_gini = None, None, float('inf')
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                
                gini = self._gini_impurity(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }
    
    def _gini_impurity(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """Calculate weighted Gini impurity"""
        def gini(y: np.ndarray) -> float:
            if len(y) == 0:
                return 0
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)
        
        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right
        return (n_left / n_total) * gini(left_y) + (n_right / n_total) * gini(right_y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained tree
        
        Args:
            X: Input features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        X = np.array(X)
        return np.array([self._predict_sample(x, self.tree) for x in X])
    
    def _predict_sample(self, x: np.ndarray, node: dict) -> Union[str, int]:
        """
        Predict single sample
        
        Args:
            x: Single sample features
            node: Current tree node
            
        Returns:
            Union[str, int]: Predicted class
        """
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])


class RandomForest:
    """Random Forest Classifier"""
    
    def __init__(
        self, 
        n_trees: int = 10, 
        max_depth: int = 10, 
        min_samples_split: int = 2, 
        max_features: str = 'sqrt'
    ):
        """
        Initialize Random Forest
        
        Args:
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples to split a node
            max_features: Number of features for each tree
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train random forest
        
        Args:
            X: Input features
            y: Target labels
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        if self.max_features == 'sqrt':
            n_features_per_tree = int(np.sqrt(n_features))
        else:
            n_features_per_tree = n_features
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Random feature selection
            feature_indices = np.random.choice(n_features, n_features_per_tree, replace=False)
            X_subset = X_bootstrap[:, feature_indices]
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_subset, y_bootstrap)
            self.trees.append((tree, feature_indices))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using majority voting
        
        Args:
            X: Input features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        X = np.array(X)
        all_predictions = []
        
        for tree, feature_indices in self.trees:
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        final_predictions = []
        
        for i in range(len(X)):
            votes = all_predictions[:, i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        
        return np.array(final_predictions)


class LogisticRegression:
    """Logistic Regression for binary classification"""
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        """
        Initialize Logistic Regression
        
        Args:
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train logistic regression using gradient descent
        
        Args:
            X: Input features
            y: Target labels (binary)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Input features to predict
            
        Returns:
            numpy.ndarray: Predicted labels (0 or 1)
        """
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)