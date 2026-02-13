import numpy as np
import pandas as pd


class Node:

    def __init__(self, X, y, depth, max_depth, max_features, random_state, rng=None):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.Series) else y
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None

        self.depth = depth
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.rng = rng

        self.left_leaf = None
        self.right_leaf = None
        self.split_feature = None
        self.split_value = None

        self.is_leaf = self._is_leaf_check()
        self.value = None
        if self.is_leaf:
            self.value = self._get_value()

    def _is_leaf_check(self):
        return (
            self.depth >= self.max_depth or
            len(self.y) < 2
        )

    def _get_value(self):
        return float(np.mean(self.y))

    def _sse(self, y):
        if len(y) == 0:
            return 0.0
        mean = np.mean(y)
        return np.sum((y - mean) ** 2)

    def _get_best_split(self):
        n_samples, n_features = self.X.shape
        if n_samples < 2:
            return None, None, None, None

        best_gain = -np.inf
        best_feature = None
        best_split_value = None
        best_left_idx = None
        best_right_idx = None

        parent_error = self._sse(self.y)

        feature_indices = range(n_features)
        if self.max_features == "sqrt":
            feature_indices = self.rng.choice(
                list(feature_indices),
                size=int(np.sqrt(n_features)),
                replace=False
            )
        elif isinstance(self.max_features, float) and 0 < self.max_features <= 1:
            feature_indices = self.rng.choice(
                list(feature_indices),
                size=int(n_features * self.max_features),
                replace=False
            )

        for feature_idx in feature_indices:
            feature_values = self.X[:, feature_idx]
            sorted_idx = np.argsort(feature_values)
            sorted_feature = feature_values[sorted_idx]
            sorted_y = self.y[sorted_idx]

            for i in range(1, n_samples):
                if sorted_feature[i] == sorted_feature[i - 1]:
                    continue

                left_y = sorted_y[:i]
                right_y = sorted_y[i:]

                error = self._sse(left_y) + self._sse(right_y)
                gain = parent_error - error

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_split_value = (sorted_feature[i - 1] + sorted_feature[i]) / 2
                    best_left_idx = sorted_idx[:i]
                    best_right_idx = sorted_idx[i:]

        if best_gain <= 0:
            return None, None, None, None

        X_left = self.X[best_left_idx]
        y_left = self.y[best_left_idx]
        X_right = self.X[best_right_idx]
        y_right = self.y[best_right_idx]

        self.split_feature = best_feature
        self.split_value = best_split_value

        return X_left, y_left, X_right, y_right

    def build_tree(self):
        if self.is_leaf:
            return self

        X_left, y_left, X_right, y_right = self._get_best_split()

        if X_left is None or X_right is None:
            self.is_leaf = True
            self.value = self._get_value()
            return self

        self.left_leaf = Node(
            pd.DataFrame(X_left, columns=self.feature_names) if self.feature_names else X_left,
            pd.Series(y_left) if self.feature_names else y_left,
            self.depth + 1,
            self.max_depth,
            self.max_features,
            self.random_state,
            self.rng
        ).build_tree()

        self.right_leaf = Node(
            pd.DataFrame(X_right, columns=self.feature_names) if self.feature_names else X_right,
            pd.Series(y_right) if self.feature_names else y_right,
            self.depth + 1,
            self.max_depth,
            self.max_features,
            self.random_state,
            self.rng
        ).build_tree()

        return self


class MyDecisionTreeRegressor:

    def __init__(self, max_depth=5, max_features=None, random_state=21):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def fit(self, X_train, y_train):
        rng = np.random.RandomState(self.random_state)
        root = Node(
            X_train, y_train,
            depth=0,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
            rng=rng
        )
        self.tree = root.build_tree()
        return self

    def predict(self, X_test):
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        preds = np.zeros(X_test.shape[0])

        for i, row in enumerate(X_test):
            node = self.tree
            while not node.is_leaf:
                if row[node.split_feature] > node.split_value:
                    node = node.right_leaf
                else:
                    node = node.left_leaf
            preds[i] = node.value

        return preds
    


class MyGBDTClassifier():

    def __init__(
        self, 
        max_depth,
        number_of_trees,
        max_features,
        learning_rate,
        random_state=21
        ):
        
        self.max_depth = max_depth
        self.n_of_trees = number_of_trees
        self.max_features = max_features
        self.random_state = random_state
        self.lr = learning_rate

        self.trees = [0] * number_of_trees

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def fit(self, X_train, y_train):

        X_train = X_train.values
        y_train = y_train.values

        n_of_samples = X_train.shape[0]
        predicted_logits = np.zeros(n_of_samples)

        for i in range(self.n_of_trees):
            
            p = self.sigmoid(predicted_logits)
            antigrad = - (p - y_train)
            
            tree = MyDecisionTreeRegressor(
                max_depth=self.max_depth, 
                max_features=self.max_features, 
                random_state=self.random_state+i
                ).fit(X_train, antigrad)
            
            predicted_logits += self.lr * tree.predict(X_train)
            self.trees[i] = tree

        return self

    
    def predict_proba(self, X_test):

        n_of_samples = X_test.shape[0]
        predicted_logits = np.zeros(n_of_samples)

        for i in range(self.n_of_trees):
            predicted_logits += self.lr * self.trees[i].predict(X_test)
            
        class_1_probas = self.sigmoid(predicted_logits)
        class_1_probas = class_1_probas.reshape(-1, 1)
        return np.hstack([1 - class_1_probas, class_1_probas])








