import numpy as np
import pandas as pd

class Node():
    
    def __init__(self, X, y, depth, max_depth, max_features, random_state, rng=None):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.Series) else y
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        self.depth = depth
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.rng = rng
        
        self.right_leaf = None
        self.left_leaf = None
        self.split_feature = None
        self.split_value = None
        
        self.is_leaf = self._is_leaf_check()
        self.probas = None
        if self.is_leaf:
            self.probas = self._get_probas()
    
    def _is_leaf_check(self):
        return (np.unique(self.y).shape[0] == 1 or 
                self.depth >= self.max_depth or
                len(self.y) < 2)
    
    def _get_probas(self):
        total_count = len(self.y)
        class_1_count = np.sum(self.y)
        class_0_count = total_count - class_1_count
        cl_1_share = class_1_count / total_count if total_count > 0 else 0.5
        cl_0_share = 1 - cl_1_share
        return (cl_0_share, cl_1_share)
    
    def _get_best_split(self):
        n_samples, n_features = self.X.shape
        total_count = n_samples
        
        if total_count < 2:
            return None, None, None, None
        
        class_1_total = np.sum(self.y)
        class_0_total = total_count - class_1_total
        
        best_gini_gain = -1
        best_feature = None
        best_split_value = None
        best_left_indices = None
        best_right_indices = None
        
        features_idx_list = range(n_features)
        if self.max_features == 'sqrt':
            features_idx_list = self.rng.choice(features_idx_list, size=int(n_features ** 0.5), replace=False)
        elif isinstance(self.max_features, float) and 0 < self.max_features <= 1:
            features_idx_list = self.rng.choice(features_idx_list, size=int(n_features * self.max_features), replace=False)

        for feature_idx in features_idx_list:

            feature_values = self.X[:, feature_idx]
            y_values = self.y
            
            sorted_indices = np.argsort(feature_values)
            sorted_feature = feature_values[sorted_indices]
            sorted_y = y_values[sorted_indices]
            
            left_total = 1
            left_class_1 = float(sorted_y[0])
            left_class_0 = left_total - left_class_1
            
            right_total = total_count - left_total
            right_class_1 = class_1_total - left_class_1
            right_class_0 = class_0_total - left_class_0
            
            parent_gini = self._gini_impurity(total_count, class_1_total, class_0_total)
            
            for i in range(1, total_count):
                if sorted_feature[i] != sorted_feature[i-1]:

                    left_gini = self._gini_impurity(left_total, left_class_1, left_class_0)
                    right_gini = self._gini_impurity(right_total, right_class_1, right_class_0)
                    
                    gini_gain = parent_gini - (left_total/total_count) * left_gini - (right_total/total_count) * right_gini
                    
                    if gini_gain > best_gini_gain:
                        best_gini_gain = gini_gain
                        best_feature = feature_idx
                        best_split_value = (sorted_feature[i-1] + sorted_feature[i]) / 2
                        best_left_indices = sorted_indices[:i]
                        best_right_indices = sorted_indices[i:]
                
                left_total += 1
                left_class_1 += sorted_y[i]
                left_class_0 = left_total - left_class_1
                
                right_total = total_count - left_total
                right_class_1 = class_1_total - left_class_1
                right_class_0 = class_0_total - left_class_0
        
        if best_gini_gain <= 0:
            return None, None, None, None
        
        X_right = self.X[best_right_indices]
        y_right = self.y[best_right_indices]
        X_left = self.X[best_left_indices]
        y_left = self.y[best_left_indices]
        
        self.split_feature = best_feature
        self.split_value = best_split_value
        
        return X_right, y_right, X_left, y_left
    
    def _gini_impurity(self, total_count, class_1_count, class_0_count):
        if total_count == 0:
            return 0
        cl_1_share = class_1_count / total_count
        cl_0_share = 1 - cl_1_share
        return 1 - (cl_1_share ** 2 + cl_0_share ** 2)
    
    def build_tree(self):
        if self.is_leaf:
            return self
        
        X_right, y_right, X_left, y_left = self._get_best_split()
        
        if X_right is None or X_left is None:
            self.is_leaf = True
            self.probas = self._get_probas()
            return self
        
        self.right_leaf = Node(
            pd.DataFrame(X_right, columns=self.feature_names) if self.feature_names else X_right,
            pd.Series(y_right) if self.feature_names else y_right,
            self.depth + 1, 
            self.max_depth,
            self.max_features,
            self.random_state, 
            self.rng
        ).build_tree()
        
        self.left_leaf = Node(
            pd.DataFrame(X_left, columns=self.feature_names) if self.feature_names else X_left,
            pd.Series(y_left) if self.feature_names else y_left,
            self.depth + 1,
            self.max_depth,
            self.max_features,
            self.random_state,
            self.rng
        ).build_tree()
        
        return self

class MyDecisionTreeClassifier():

    def __init__(self, max_depth=5, max_features=None, random_state=21):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None


    def fit(self, X_train, y_train):
        rng = np.random.RandomState(self.random_state)
        root = Node(
            X_train, y_train, depth=0, 
            max_depth=self.max_depth, 
            max_features=self.max_features, 
            random_state=self.random_state,
            rng=rng
            )
        self.tree = root.build_tree()
        return self

    
    def predict_proba(self, X_test):
        
        X_test = X_test.values
        n_samples = X_test.shape[0]
        probas = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            node = self.tree
            row = X_test[i]

            while not node.is_leaf:
                if row[node.split_feature] > node.split_value:
                    node = node.right_leaf
                else:
                    node = node.left_leaf
            probas[i] = node.probas

        return probas
                

    def predict(self, X_test):
        proba = self.predict_proba(X_test)
        predicted = (proba[:, 1] > 0.5).astype(int)     # samples with probability > 0.5 are marked as class 1
        return predicted
    

    def print_tree(self):
        def _print_node(node, prefix="", is_right_child=False, depth=0):
            if node is None:
                return
            direction = ""
            if depth > 0:
                if is_right_child:
                    direction = "R-> "
                else:
                    direction = "L-> "
            if node.is_leaf:
                node_text = f"Leaf: probas={node.probas}"
                node_text += f" (samples={len(node.y)})"
            else:
                node_text = f"[{node.feature_names[node.split_feature]} > {node.split_value:.3f}]"
                node_text += f" (samples={len(node.y)})"
            print(prefix + direction + node_text)
            
            if not node.is_leaf:
                child_prefix = prefix + "    "
                _print_node(node.right_leaf, child_prefix, is_right_child=True, depth=depth+1)
                _print_node(node.left_leaf, child_prefix, is_right_child=False, depth=depth+1)
        
        _print_node(self.tree, depth=0)


class ExtraRandomizedNode(Node):

    def __init__(self, X, y, depth, max_depth, max_features, random_state, rng=None):
        super().__init__(X, y, depth=depth, max_depth=max_depth, max_features=max_features, random_state=random_state)
        self.rng = rng

    def build_tree(self):

        if self.is_leaf:
            return self
        
        X_right, y_right, X_left, y_left = self._get_best_split()
        
        if X_right is None or X_left is None:
            self.is_leaf = True
            self.probas = self._get_probas()
            return self
        
        self.right_leaf = ExtraRandomizedNode(
            pd.DataFrame(X_right, columns=self.feature_names) if self.feature_names else X_right,
            pd.Series(y_right) if self.feature_names else y_right,
            self.depth + 1,
            self.max_depth,
            self.max_features,
            self.random_state,
            self.rng
        ).build_tree()
        
        self.left_leaf = ExtraRandomizedNode(
            pd.DataFrame(X_left, columns=self.feature_names) if self.feature_names else X_left,
            pd.Series(y_left) if self.feature_names else y_left,
            self.depth + 1,
            self.max_depth,
            self.max_features,
            self.random_state,
            self.rng
        ).build_tree()
        
        return self


    def _get_best_split(self):

        n_samples, n_features = self.X.shape
        total_count = n_samples
        
        if total_count < 2:
            return None, None, None, None
        
        class_1_total = np.sum(self.y)
        class_0_total = total_count - class_1_total
        parent_gini = self._gini_impurity(total_count, class_1_total, class_0_total)
        
        best_gini_gain = -1
        best_feature = None
        best_split_value = None
        best_left_indices = None
        best_right_indices = None

        features_idx_list = range(n_features)
        if self.max_features == 'sqrt':
            features_idx_list = self.rng.choice(features_idx_list, size=int(n_features ** 0.5), replace=False)
        elif isinstance(self.max_features, float) and 0 < self.max_features <= 1:
            features_idx_list = self.rng.choice(features_idx_list, size=int(n_features * self.max_features), replace=False)
        
        for feature_idx in features_idx_list:
            feature_values = self.X[:, feature_idx]
            
            random_idx = self.rng.randint(0, n_samples)
            split_value = feature_values[random_idx]
            
            right_mask = feature_values > split_value
            left_mask = ~right_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            y_left = self.y[left_mask]
            y_right = self.y[right_mask]
            
            left_total = len(y_left)
            right_total = len(y_right)
            
            left_class_1 = np.sum(y_left)
            left_class_0 = left_total - left_class_1
            
            right_class_1 = np.sum(y_right)
            right_class_0 = right_total - right_class_1
            
            left_gini = self._gini_impurity(left_total, left_class_1, left_class_0)
            right_gini = self._gini_impurity(right_total, right_class_1, right_class_0)
            
            gini_gain = parent_gini - (left_total/total_count) * left_gini - (right_total/total_count) * right_gini
            
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_feature = feature_idx
                best_split_value = split_value
                best_left_indices = np.where(left_mask)[0]
                best_right_indices = np.where(right_mask)[0]
        
        if best_gini_gain <= 0 or best_left_indices is None:
            return None, None, None, None
        
        X_right = self.X[best_right_indices]
        y_right = self.y[best_right_indices]
        X_left = self.X[best_left_indices]
        y_left = self.y[best_left_indices]
        
        self.split_feature = best_feature
        self.split_value = best_split_value
        
        return X_right, y_right, X_left, y_left
    

class MyExtraRandomizedTreeClassifier(MyDecisionTreeClassifier):

    def fit(self, X_train, y_train):
        rng = np.random.RandomState(self.random_state)
        root = ExtraRandomizedNode(
            X_train, 
            y_train, 
            depth=0, 
            max_depth=self.max_depth, 
            max_features=self.max_features,
            random_state=self.random_state,
            rng=rng
            )
        self.tree = root.build_tree()
        return self
    

from joblib import Parallel, delayed
    
    
class MyRandomForestClassifier():

    class BootstrapTree(MyDecisionTreeClassifier):
        
        def fit(self, X_train, y_train):
            n_samples = X_train.shape[0]
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X_train.index, size=n_samples, replace=True)
            X_bootstrap = X_train.iloc[idx].reset_index(drop=True)
            y_bootstrap = y_train.iloc[idx].reset_index(drop=True)
            return super().fit(X_bootstrap, y_bootstrap)

    
    def __init__(self, max_depth=10, number_of_trees=10, max_features='sqrt', random_state=21, n_jobs=-1):
        self.max_depth = max_depth
        self.number_of_trees = number_of_trees
        self.max_features = max_features
        self.random_state = random_state
        
        self.n_jobs = n_jobs

        self.trees = [0] * number_of_trees

    # def fit(self, X_train, y_train):
        
    #     for i in range(self.number_of_trees):
    #         self.trees[i] = self.BootstrapTree(
    #             max_depth=self.max_depth,
    #             max_features=self.max_features,
    #             random_state=self.random_state + i
    #         ).fit(X_train, y_train)
    #         print(f"tree n. {i} fit complete")
    #     return self

    def fit(self, X_train, y_train):

        def _fit_single_tree(tree_idx):

            tree = self.BootstrapTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state + tree_idx
            )
            fitted_tree = tree.fit(X_train, y_train)
            return fitted_tree
        
        print(f"Training {self.number_of_trees} trees with n_jobs={self.n_jobs}:")
        
        self.trees = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(_fit_single_tree)(i) 
            for i in range(self.number_of_trees)
        )
        
        return self


    # def predict_proba(self, X_test):
    #     probas = self.trees[0].predict_proba(X_test)
    #     for tree in self.trees[1:]:
    #         probas = probas + tree.predict_proba(X_test)

    #     probas = probas / self.number_of_trees
    #     return probas
    
    def predict_proba(self, X_test):
        all_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda t: t.predict_proba(X_test))(tree)
            for tree in self.trees
        )
        return np.mean(all_probas, axis=0)


    def predict(self, X_test):
        proba = self.predict_proba(X_test)
        predicted = (proba[:, 1] > 0.5).astype(int)     # samples with probability > 0.5 are marked as class 1
        return predicted

        
class MyExtraTreesClassifier(MyRandomForestClassifier):

    def fit(self, X_train, y_train):
        
        def _fit_single_tree(tree_idx):

            tree = MyExtraRandomizedTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state + tree_idx
            )
            fitted_tree = tree.fit(X_train, y_train)
            return fitted_tree
        
        print(f"Training {self.number_of_trees} trees with n_jobs={self.n_jobs}:")
        
        self.trees = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(_fit_single_tree)(i) 
            for i in range(self.number_of_trees)
        )
        
        return self