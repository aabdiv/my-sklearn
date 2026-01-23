import numpy as np
import pandas as pd


def my_roc_auc_score(y_true, y_score):
    """
    calculate Area Under the ROC Curve (ROC AUC) for binary classification
    
    y_true: array with true class labels
    y_score: array with probabilities of class 1 (positive) got from .predict_proba()[:, 1]
    """

    argsorted = y_score.argsort()[::-1]     # sort probabilitites descendingly - we start with (TPR, FPR) at (0,0) and add positive labels progressively

    tpr_list = [0]
    fpr_list = [0]              # here TPR and FPR are stored after every new boundary

    fn = np.sum(y_true)         # at first everything is labeled as 0, so all actual 1's go to FN
    tn = len(y_true) - fn       # all actual 0's go to TN
    tp = 0
    fp = 0

    for idx in argsorted:
        
        if y_true.iloc[idx] == 1:    # adjust confusion matrix according to the true label 
            tp += 1
            fn -= 1
        else:
            tn -= 1
            fp += 1

        tpr = tp / (tp + fn)        # calculate True Positive Rate and
        fpr = fp / (fp + tn)        # False Positive Rate

        if fpr != fpr_list[-1]:     # if incurred increase in FPR (graph moved right)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        else:                       # if managed to increase TPR staying at the same FPR (i.e. hit correct positive, move upwards on graph)
            tpr_list[-1] = tpr

    bases = np.array([abs(curr - next) for curr, next in zip(fpr_list, fpr_list[1:])])        # calculate bases for all trapezoids in TPR-FPR graph
    heights = np.array([(curr + next) / 2 for curr, next in zip(tpr_list, tpr_list[1:])])     # trapezoids heights

    auc = np.sum(bases * heights)      # sum all areas

    return auc    # return area under the curve


def my_gini(y_true, y_score):
    return 2 * my_roc_auc_score(y_true, y_score) - 1


def my_precision_score(y_true, y_pred):
    tp = (y_true & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    return tp / (tp + fp)

def my_recall_score(y_true, y_pred):
    tp = (y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    return tp / (tp + fn)

def my_f1_score(y_true, y_pred):
    prec = my_precision_score(y_true, y_pred)
    recall = my_recall_score(y_true, y_pred)
    return 2 * prec * recall / (prec + recall)

def my_pr_auc_score(y_true, y_score):
    """
    calculate Area Under the Precision-Recall Curve (PR AUC) for binary classification
    
    y_true: array with true class labels
    y_score: array with probabilities of class 1 (positive) got from .predict_proba()[:, 1]
    """

    argsorted = y_score.argsort()[::-1]     # sort probabilitites descendingly - we start with (Precision, Recall) at (0,0) and add positive labels progressively

    precision_list = []        # here precision and recall are stored after every new boundary,
    recall_list = [0]           # zero added to recall list, since recall form base of trapezoids for AUC calculation

    fn = np.sum(y_true)         # at first everything is labeled as 0, so all actual 1's go to FN
    tn = len(y_true) - fn       # all actual 0's go to TN
    tp = 0
    fp = 0

    for idx in argsorted:
        
        if y_true.iloc[idx] == 1:      # adjust confusion matrix according to the true label 
            tp += 1
            fn -= 1
        else:
            tn -= 1
            fp += 1

        precision = tp / (tp + fp)    # calculate precision and
        recall = tp / (tp + fn)       # recall

        precision_list.append(precision)
        recall_list.append(recall)

        # if not precision_list:        # first threshold
        #     precision_list.append(precision)
        #     recall_list.append(recall)
        #     continue

        # if recall == recall_list[-1] and precision < precision_list[-1]:   # if incurred precision loss staying on the same recall 
        #     continue                                                       # (i.e. hit false positive, moved downwards on graph)
        # else:
        #     precision_list.append(precision)
        #     recall_list.append(recall) 

    bases = np.array([abs(curr - next) for curr, next in zip(recall_list, recall_list[1:])])        # calculate bases for all trapezoids in Precision-Recall graph
    heights = np.array(precision_list)     # trapezoids heights

    auc = np.sum(bases * heights)      # sum all areas

    return auc    # return area under the curve


class MyLogisticRegression():
    """
    Custom logistic regression implementation with SGD optimization.
    
    Parameters
    ----------
    batch_size : int, default=32
        Batch size for SGD (ignored for other methods)
    max_iter : int, default=1000
        Maximum number of iterations for gradient-based methods
    tol : float, default=0.01
        Tolerance for early stopping (convergence threshold)
    eta : float, default=0.01
        Learning rate for gradient-based methods
    random_state : int, default=21
        Random seed for reproducible deterministic SGD
    log_freq : int, default=100
        Epoch loss logging interval
    weights_init : str, default='zeros'
        Method for weights initialization before SGD (zeros or random)
    
    Attributes
    ----------
    coef_ : array
        Feature coefficients (weights for input features)
    intercept_ : float
        Bias term (intercept)
    n_iter_ : int
        Actual number of iterations performed during fit
    loss_by_epoch : array[float]
        Total Loss calculated after every SGD epoch 
    """
    
    def __init__(self, batch_size=256, max_iter=100, tol=0.001, eta=0.1, random_state=21, log_freq=2, weights_init='zeros'):
        
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta  
        self.random_state = random_state
        self.weights_init = weights_init
        self.method = 'SGD'

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

        self.loss_by_epoch = None
        self.log_freq = log_freq


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    
    def fit(self, X_train, y_train):
        """
        Fit logistic regression model to training data.
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self for method chaining
        """
        
        np.random.seed(self.random_state)    # for deterministic SGD
        n, d = X_train.shape                 # n samples, d features 
        
        if self.weights_init == 'zeros':
            w = np.zeros(d + 1)              # initialize weights 
        elif self.weights_init == 'random':
            w = np.random.random(d + 1)

        if isinstance(X_train, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train])        # add 1 as a first element of sample vectors for operations with weights vector having bias term

        iters_no_change = 0         # to track if loss function does not decrease for a number consecutive epochs (converged)
        self.n_iter_ = self.max_iter
        steps_per_epoch = np.ceil(n / self.batch_size)      # number of batches in the X_train

        self.loss_by_epoch = np.zeros(int(self.max_iter * steps_per_epoch))     # inititalize array for loss calculation after every batch

        if self.method == 'SGD':
        
            for E in range(self.max_iter):          # epoch

                index = np.random.permutation(n)    # shuffle indices for SGD
                
                for step, i in enumerate(range(0, n, self.batch_size)):
                    X = X_with_bias[index[i:i+self.batch_size]]    # get batch
                    y = y_train.values[index[i:i+self.batch_size]]

                    val = X.dot(w)                 # logit
                    probs = self._sigmoid(val)     # calculate probabilities from logits
                    errors = y - probs

                    grad = X.T @ errors / -self.batch_size         # calculate gradient dL/dw
                    
                    w -= self.eta * grad            # update weights

                    total_val = X_with_bias.dot(w)  # recalculate logits using updated w for loss calculation
                    total_probs = self._sigmoid(total_val)

                    epoch_loss = -1/n * np.sum((y_train * np.log(total_probs) + (1 - y_train) * np.log(1-total_probs)))
                    self.loss_by_epoch[int(E * steps_per_epoch + step)] = epoch_loss        # put current loss in array


                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if self.log_freq != 0 and E % self.log_freq == 0:    # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")

        self.loss_by_epoch = np.trim_zeros(self.loss_by_epoch)

        self.intercept_ = w[0]
        self.coef_ = w[1:]


        return self


    def predict_proba(self, X_test):
        """
        Predict class probabilities for test data.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples, 2)
            Predicted class probabilities, class 0 probability first, class 1 probability second
        """
        if self.coef_ is None:
            raise Exception('fit first')

        n = X_test.shape[0]   # n predictions
        if isinstance(X_test, pd.DataFrame):        # add 1 as a first element of sample vectors for operations with weights vector having bias term
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test])

        w = np.hstack([self.intercept_, self.coef_])      # combine intercept (bias) term with feature coefficients

        val = X_with_bias.dot(w)                    # logits
        class_1_probs = self._sigmoid(val)          # class 1 probabilities
        class_0_probs = 1 - class_1_probs

        proba = np.vstack([class_0_probs, class_1_probs]).T
        
        return proba
    

    def predict(self, X_test):
        """
        Predict class labels for test data with 0.5 decision threshold
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples)
            Predicted class labels
        """
        if self.coef_ is None:
            raise Exception('fit first')

        proba = self.predict_proba(X_test=X_test)

        predicted = (proba[:, 1] > 0.5).astype(int)         # samples with probability > 0.5 are marked as class 1

        return predicted
    


class MyKNNCLassifier():
    """
    Custom KNN classifier implementation using Euclidean distance.
    X_train and y_train are simply stored in the attributes for further predicts
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use

    Attributes
    ----------
    X_train_ : array-like, shape (n_samples, n_features)
        Training features
    y_train_ : array-like, shape (n_samples,)
        Target values
    """
    
    def _validate_input(self, n_neighbors):
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors <= 0:
            raise ValueError('n_neighbors must a positive integer')


    def __init__(self, n_neighbors=5):

        self._validate_input(n_neighbors=n_neighbors)

        self.n_neighbors = n_neighbors

        self.X_train_ = None
        self.y_train_ = None

    
    def fit(self, X_train, y_train):
        """
        Fit KNN classifier to training data.
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self for method chaining
        """

        if isinstance(X_train, np.ndarray):
            self.X_train_ = X_train
        else:
            self.X_train_ = X_train.values

        if isinstance(y_train, np.ndarray):
            self.y_train_ = y_train
        else:
            self.y_train_ = y_train.values

        return self

        
    # def predict_proba(self, X_test):
    #     """
    #     Predict class probabilities for test data.
        
    #     Parameters
    #     ----------
    #     X_test : array-like, shape (n_samples, n_features)
    #         Test features
            
    #     Returns
    #     -------
    #     y_pred : array, shape (n_samples, 2)
    #         Predicted class probabilities, class 0 probability first, class 1 probability second
    #     """
    #     if self.X_train_ is None or self.y_train_ is None:
    #         raise Exception('fit first')

    #     n = X_test.shape[0]   # n predictions
    #     m, d = self.X_train_.shape      # m vectors in train, d features

    #     class_1_probs = np.zeros(n)

    #     for row_i, test_row in enumerate(X_test.values):

    #         distances = np.zeros(m)

    #         for idx, train_row in enumerate(self.X_train_):
    #             distances[idx] = np.sqrt(np.sum(np.square(test_row - train_row)))

    #         neighbors_distances = distances.copy()[:self.n_neighbors]
    #         neighbors_idx = np.arange(self.n_neighbors)

    #         for idx, distance in enumerate(distances):
    #             if idx >= self.n_neighbors:
    #                 max_dist_idx = np.argmax(neighbors_distances)
    #                 if distance < neighbors_distances[max_dist_idx]:
    #                     neighbors_distances[max_dist_idx] = distance
    #                     neighbors_idx[max_dist_idx] = idx

    #         neighbors_classes = self.y_train_[neighbors_idx]

    #         class_1_probs[row_i] = np.sum(neighbors_classes) / self.n_neighbors

    #     class_0_probs = 1 - class_1_probs

    #     proba = np.vstack([class_0_probs, class_1_probs]).T
        
    #     return proba
    

    def predict_proba(self, X_test):
        """
        Predict class probabilities for test data.
        Squared distances are used for sorting,
        squared Euclidean norm = (x - y)^2 = |x|^2 - 2xy + |y|^2
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples, 2)
            Predicted class probabilities, class 0 probability first, class 1 probability second
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise Exception('fit first')
        
        if isinstance(X_test, np.ndarray):
            X_test_np = X_test.copy()
        else:
            X_test_np = X_test.values.copy()

        test_squared_norm = np.sum(X_test_np ** 2, axis=1)          # ||X||^2 where X is matrix of test vectors, size is (n_test_samples, 1)
        train_squared_norm = np.sum(self.X_train_ ** 2, axis=1)     # ||Y||^2 where Y is matrix of train vectors, size is (n_train_samples, 1)

        dot_product = self.X_train_ @ X_test.T                      # x @ y term, size is (n_train_samples, n_test_samples)

        distance_matrix = train_squared_norm.reshape(-1, 1) - 2 * dot_product + test_squared_norm           # squared distances, ij element is squared distance between ith train vector and jth test vector

        n_closest = np.argpartition(distance_matrix, kth=self.n_neighbors-1, axis=0)[:self.n_neighbors, :]  # indices of n closest train vectors for jth test vector, size is (n_neighbors, n_test_samples)

        closest_classes = self.y_train_[n_closest]          # get classes of n closest train vectors, size is (n_neighbors, n_test_samples)

        class_1_probs = np.mean(closest_classes, axis=0)    # for binary classification with classes (0, 1) mean gets the probability of class 1 
        class_0_probs = 1 - class_1_probs

        proba = np.vstack([class_0_probs, class_1_probs]).T
        
        return proba


    def predict(self, X_test):
        """
        Predict class labels for test data with 0.5 decision threshold
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples)
            Predicted class labels
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise Exception('fit first')

        proba = self.predict_proba(X_test=X_test)

        predicted = (proba[:, 1] > 0.5).astype(int)     # samples with probability > 0.5 are marked as class 1

        return predicted


class MyMixedNB():
    """
    Custom Mixed Naive Bayes classifier implementation.
    Automatically detects binary/numerical features and uses Bernoulli/Gaussian likelihoods accordingly.

    Parameters
    ----------

    Attributes
    ----------
    X_train_ : array-like, shape (n_samples, n_features)
        Training features
    y_train_ : array-like, shape (n_samples,)
        Target values
    """


    def __init__(self):

        self.binary_features_idx = None
        self.numerical_features_idx = None

        self.bin_likelihoods_cl1 = None
        self.bin_likelihoods_cl0 = None

        self.num_mu_cl1 = None
        self.num_mu_cl0 = None
        self.num_sigma_cl1 = None
        self.num_sigma_cl0 = None

        self.prior_class1 = None
        self.prior_class0 = None


    def fit(self, X_train, y_train):
        """
        Fit MixedNB classifier to training data.
        Class priors are estimated using classes shares in total number of train samples from y_train.
        Likelihoods for binaries / mu and sigma for numericals are stored in attributes after .fit()
        For numerical stability, 1 and 2 are added to numerator and denumerator respectively when calculating binary likelihoods
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self for method chaining
        """

        if isinstance(X_train, np.ndarray):
            X_train_np = X_train.copy()
        else:
            X_train_np = X_train.values.copy()

        if isinstance(y_train, np.ndarray):
            y_train_np = y_train.copy()
        else:
            y_train_np = y_train.values.copy()

        binary_features_mask = np.isin(X_train_np, [0, 1]).all(axis=0)
        self.binary_features_idx = np.where(binary_features_mask)[0]        # indices of columns with binary features
        self.numerical_features_idx = np.where(~binary_features_mask)[0]    # indices of columns with numerical features

        X_train_binary = X_train_np[:, self.binary_features_idx]            # binary features matrix from X_train
        X_train_numerical = X_train_np[:, self.numerical_features_idx]      # numerical features matrix from X_train

        y_class_1_idx = np.where(y_train.astype(np.bool))[0]                # indices of samples with true class 1
        y_class_0_idx = np.where(~y_train.astype(np.bool))[0]               # indices of samples with true class 0

        X_train_binary_cl1 = X_train_binary[y_class_1_idx, :].copy()        # matrix of binary features for class 1 samples
        X_train_binary_cl0 = X_train_binary[y_class_0_idx, :].copy()        # matrix of binary features for class 0 samples
        X_train_numerical_cl1 = X_train_numerical[y_class_1_idx, :].copy()  # matrix of numerical features for class 1 samples
        X_train_numerical_cl0 = X_train_numerical[y_class_0_idx, :].copy()  # matrix of numerical features for class 0 samples

        self.bin_likelihoods_cl1 = (np.sum(X_train_binary_cl1, axis=0) + 1) / (len(X_train_binary_cl1) + 2)      # Laplace smoothing for numerical stability
        self.bin_likelihoods_cl0 = (np.sum(X_train_binary_cl0, axis=0) + 1) / (len(X_train_binary_cl0) + 2)      # likelihoods P(X|C) of binary features for C=0 (class 0)

        self.num_mu_cl1 = np.mean(X_train_numerical_cl1, axis=0)            # means of numerical features for class 1 for Normal distribution estimation
        self.num_mu_cl0 = np.mean(X_train_numerical_cl0, axis=0)            # means of numerical features for class 0
        self.num_sigma_cl1 = np.std(X_train_numerical_cl1, axis=0)          # standard deviations of numerical features for class 1
        self.num_sigma_cl0 = np.std(X_train_numerical_cl0, axis=0)          # standard deviations of numerical features for class 0

        self.prior_class1 = len(y_class_1_idx) / len(y_train_np)            # priors p(C) for C=1 (class 1)
        self.prior_class0 = len(y_class_0_idx) / len(y_train_np)            # priors p(C) for C=0 (class 0)

        return self


    def _gaussian_pdf_numpy(self, x, mu, sigma):
        """
        Calculates the Gaussian PDF for a given x, mean (mu), and 
        standard deviation (sigma) using NumPy operations.
        """
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
        return pdf
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for test data.
        For predict_proba probabilities, posterior proportional estimates P(X|C)*P(C) are rescaled using their sum: 
        P(1|X) = P(X|1)*P(1) / (P(X|1)*P(1) + P(X|0)*P(0)),
        where
        P(1|X) - posterior
        P(X|1) - likelihood
        P(1) - prior

        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples, 2)
            Predicted class probabilities, class 0 probability first, class 1 probability second
        """
        if self.bin_likelihoods_cl0 is None or self.bin_likelihoods_cl1 is None:
            raise Exception('fit first')
        
        if isinstance(X_test, np.ndarray):
            X_test_np = X_test.copy()
        else:
            X_test_np = X_test.values.copy()

        X_test_binary = X_test_np[:, self.binary_features_idx]                  # matrix of binary features from X_test
        X_test_numerical = X_test_np[:, self.numerical_features_idx]            # matrix of numerical features from X_test

        bin_likelihoods_1 = X_test_binary * self.bin_likelihoods_cl1 + (1 - X_test_binary) * (1 - self.bin_likelihoods_cl1)     # matrix of vectors of likelihoods for class 1 = P(x|C=1)
        bin_likelihoods_0 = X_test_binary * self.bin_likelihoods_cl0 + (1 - X_test_binary) * (1 - self.bin_likelihoods_cl0)     # matrix of vectors of likelihoods for class 0 = P(x|C=0)

        class_1_probs = np.prod(bin_likelihoods_1, axis=1)          # product of likelihoods of binary featues for class 1
        class_0_probs = np.prod(bin_likelihoods_0, axis=1)          # product of likelihoods of binary featues for class 0

        for feature_column, mu1, sigma1, mu0, sigma0 in zip(X_test_numerical.T, self.num_mu_cl1, self.num_sigma_cl1, self.num_mu_cl0, self.num_sigma_cl0):
            class_1_probs = class_1_probs * self._gaussian_pdf_numpy(feature_column, mu=mu1, sigma=sigma1)      # update likelihoods via multiplying by numerical feature probability density estimated with mean and std from train
            class_0_probs = class_0_probs * self._gaussian_pdf_numpy(feature_column, mu=mu0, sigma=sigma0)

        class_1_probs = class_1_probs * self.prior_class1           # resulting estimates of posterior probability
        class_0_probs = class_0_probs * self.prior_class0           # (only estimate, since it is proportional and used for comparison between classes posteriors)

        class_1_probs_adj = class_1_probs / (class_1_probs + class_0_probs)      # posterior proportional estimates P(X|C)*P(C) are rescaled using their sum: 
        class_0_probs_adj = class_0_probs / (class_1_probs + class_0_probs)      # P(1|X) = P(X|1)*P(1) / (P(X|1)*P(1) + P(X|0)*P(0))

        proba = np.vstack([class_0_probs_adj, class_1_probs_adj]).T
        
        return proba


    def predict(self, X_test):
        """
        Predict class labels for test data with 0.5 decision threshold
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples)
            Predicted class labels
        """
        if self.bin_likelihoods_cl0 is None or self.bin_likelihoods_cl1 is None:
            raise Exception('fit first')

        proba = self.predict_proba(X_test=X_test)

        predicted = (proba[:, 1] > 0.5).astype(int)     # samples with probability > 0.5 are marked as class 1

        return predicted












        


# def my_roc_auc_score(y_true, y_score):
#     """
#     calculate Area Under the ROC Curve (ROC AUC) for binary classification
    
#     y_true: array with true class labels
#     y_score: array with probabilities of class 1 (positive) got from .predict_proba()[:, 1]
#     """

#     actual_true_sum = np.sum(y_true)
#     actual_false_sum = len(y_true) - actual_true_sum

#     argsorted = y_score.argsort()[::-1]     # sort probabilitites descendingly - we start with (TPR, FPR) at (0,0) and add positive labels progressively

#     predicted_by_boundary = np.zeros(len(y_true))    # our hard-label prediction at every possible boundary (incremental by one sample using argsorted)

#     tpr_list = [0]
#     fpr_list = [0]      # here TPR and FPR are stored after every new boundary

#     for idx in argsorted:
#         predicted_by_boundary[idx] = 1      # move classification boundary by additionally classifying one element as positive 
        
#         tp = ((predicted_by_boundary == 1) & (y_true == 1)).sum()
#         fn = ((predicted_by_boundary == 0) & (y_true == 1)).sum()
#         fp = ((predicted_by_boundary == 1) & (y_true == 0)).sum()
#         tn = ((predicted_by_boundary == 0) & (y_true == 0)).sum()
#         tpr = tp / (tp + fn)
#         fpr = fp / (fp + tn)

#         if fpr != fpr_list[-1]:     # if incurred increase in FPR (graph moved right)
#             fpr_list.append(fpr)
#             tpr_list.append(tpr)
#         else:                       # if managed to increase TPR staying at the same FPR (i.e. hit correct positive, move upwards on graph)
#             tpr_list[-1] = tpr

#     bases = np.array([abs(curr - next) for curr, next in zip(fpr_list, fpr_list[1:])])        # calculate bases for all trapezoids in TPR-FPR graph
#     heights = np.array([(curr + next) / 2 for curr, next in zip(tpr_list, tpr_list[1:])])     # trapezoids heights

#     auc = np.sum(bases * heights)      # sum all areas

#     return auc    # return area under the curve


