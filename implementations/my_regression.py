import numpy as np
import pandas as pd


class MyLinearRegressor():
    """
    Custom linear regression implementation with multiple optimization methods.
    
    Supports three training methods:
    - SGD: Stochastic Gradient Descent with mini-batches
    - GD: Full-batch Gradient Descent 
    - analytical: Closed-form solution using normal equations
    
    Parameters
    ----------
    method : str, default='SGD'
        Optimization method: 'SGD', 'GD', or 'analytical'
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
    
    Attributes
    ----------
    coef_ : array
        Feature coefficients (weights for input features)
    intercept_ : float
        Bias term (intercept)
    n_iter_ : int
        Actual number of iterations performed during fit
    """
    
    def __init__(self, method='SGD', batch_size=32, max_iter=1000, tol=0.1, eta=0.01, random_state=21, log_freq=100):
        
        self.method = method
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta  
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

        self.log_freq = log_freq

    
    def fit(self, X_train, y_train):
        """
        Fit linear regression model to training data.
        
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
        n, d = X_train.shape    # n samples, d features 
        w = np.zeros(d + 1)    # initialize weights 
        if isinstance(X_train, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train])    # add 1 as a first element of sample vectors for operations with weights vector having bias term

        iters_no_change = 0      # to track if loss function does not decrease for a number consecutive epochs (converged)
        self.n_iter_ = self.max_iter

        if self.method == 'SGD':
        
            for E in range(self.max_iter):    # epoch

                epoch_loss = 0
                index = np.random.permutation(n)    # shuffle indices for SGD
                
                for i in range(0, n, self.batch_size):
                    X = X_with_bias[index[i:i+self.batch_size]]    # get batch
                    y = y_train.values[index[i:i+self.batch_size]]
                    grad = (2 / self.batch_size) * X.T @ (X @ w - y)    # calculate gradient dL/dw
                    
                    w -= self.eta * grad    # update weights
    
                    epoch_loss += np.sum((X @ w - y) ** 2)

                epoch_loss /= n

                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")


        if self.method == 'GD':

            X = X_with_bias
            y = y_train.values
        
            for E in range(self.max_iter):
            
                grad = 2 * X.T @ (X @ w - y) / n     # gradient using the whole dataset
                    
                w -= self.eta * grad    # update weights

                epoch_loss = np.sum((X @ w - y) ** 2) / n
                
                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")


        if self.method == 'analytical':

            X = X_with_bias
            y = y_train.values

            w = np.linalg.solve(X.T @ X, X.T @ y)

        self.intercept_ = w[0]
        self.coef_ = w[1:]

        return self


    def predict(self, X_test):
        """
        Predict target values for test data.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values
        """
        if self.coef_ is None:
            raise Exception('fit first')

        n = X_test.shape[0]   # n predictions
        if isinstance(X_test, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test])

        w = np.hstack([self.intercept_, self.coef_])      # combine intercept (bias) term with feature coefficients
        
        return X_with_bias @ w
    


def r_squared(y_true, y_pred):

    y_mean = y_true.mean()

    sse = np.sum((y_pred - y_true) ** 2)

    sst =  np.sum((y_true - y_mean) ** 2)

    return 1 - (sse / sst)



class MyRidge():
    """
    Custom Ridge Regression implementation with multiple optimization methods.
    
    Supports three training methods:
    - SGD: Stochastic Gradient Descent with mini-batches
    - GD: Full-batch Gradient Descent 
    - analytical: Closed-form solution using normal equations
    
    Parameters
    ----------
    method : str, default='SGD'
        Optimization method: 'SGD', 'GD', or 'analytical'
    alpha : float, default=1.0
        Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).
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


    Attributes
    ----------
    coef_ : array
        Feature coefficients (weights for input features)
    intercept_ : float
        Bias term (intercept)
    n_iter_ : int
        Actual number of iterations performed during fit
    """
    
    def __init__(self, method='SGD', batch_size=32, max_iter=1000, tol=0.1, eta=0.01, random_state=21, log_freq=100, alpha=1.0):
        
        self.method = method
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta  
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

        self.log_freq = log_freq

    
    def fit(self, X_train, y_train):
        """
        Fit Ridge regression model to training data.
        
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
        n, d = X_train.shape    # n samples, d features 
        w = np.zeros(d + 1)    # initialize weights 
        if isinstance(X_train, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train])    # add 1 as a first element of sample vectors for operations with weights vector having bias term

        iters_no_change = 0      # to track if loss function does not decrease for a number consecutive epochs (converged)
        self.n_iter_ = self.max_iter

        if self.method == 'SGD':
        
            for E in range(self.max_iter):    # epoch

                epoch_loss = 0
                index = np.random.permutation(n)    # shuffle indices for SGD
                
                for i in range(0, n, self.batch_size):
                    X = X_with_bias[index[i:i+self.batch_size]]    # get batch
                    y = y_train.values[index[i:i+self.batch_size]]
                    grad = (2 / self.batch_size) * X.T @ (X @ w - y) + 2 * self.alpha * w  # calculate gradient dL/dw with L2 term
                    
                    w -= self.eta * grad    # update weights
    
                    epoch_loss += np.sum((X @ w - y) ** 2)

                epoch_loss /= n

                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")


        if self.method == 'GD':

            X = X_with_bias
            y = y_train.values
        
            for E in range(self.max_iter):

                grad = 2 * X.T @ (X @ w - y) / n + 2 * self.alpha * w    # gradient using the whole dataset with L2 term
                    
                w -= self.eta * grad    # update weights

                epoch_loss = np.sum((X @ w - y) ** 2) / n
                
                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")


        if self.method == 'analytical':

            X = X_with_bias
            y = y_train.values

            if self.alpha == 0:
                w, _, _, _ = np.linalg.lstsq(X, y, rcond=-1)
            else:
                w = np.linalg.solve(X.T @ X + self.alpha * np.eye(d + 1), X.T @ y)

        self.intercept_ = w[0]
        self.coef_ = w[1:]

        return self


    def predict(self, X_test):
        """
        Predict target values for test data.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values
        """
        if self.coef_ is None:
            raise Exception('fit first')

        n = X_test.shape[0]   # n predictions
        if isinstance(X_test, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test]) 

        w = np.hstack([self.intercept_, self.coef_])      # combine intercept (bias) term with feature coefficients
        
        return X_with_bias @ w



class MyLasso():
    """
    Custom Lasso Regression implementation with multiple optimization methods.
    
    Supports three training methods:
    - SGD: Stochastic Gradient Descent with mini-batches
    - GD: Full-batch Gradient Descent 
    
    Parameters
    ----------
    method : str, default='SGD'
        Optimization method: 'SGD' or 'GD'
    alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).
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


    Attributes
    ----------
    coef_ : array
        Feature coefficients (weights for input features)
    intercept_ : float
        Bias term (intercept)
    n_iter_ : int
        Actual number of iterations performed during fit
    """
    
    def __init__(self, method='SGD', batch_size=32, max_iter=1000, tol=0.1, eta=0.01, random_state=21, log_freq=100, alpha=1.0):
        
        self.method = method
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta  
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

        self.log_freq = log_freq


    def _soft_threshold(self, x, threshold):    # for ISTA
            return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X_train, y_train):
        """
        Fit Lasso regression model to training data.
        Iterative Shrinking-Threshholding Algorithm (ISTA) is used for optimization with non-smooth L1 term 
        
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
        n, d = X_train.shape    # n samples, d features 
        w = np.zeros(d + 1)    # initialize weights 
        if isinstance(X_train, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train])    # add 1 as a first element of sample vectors for operations with weights vector having bias term

        iters_no_change = 0      # to track if loss function does not decrease for a number consecutive epochs (converged)
        self.n_iter_ = self.max_iter

        if self.method == 'SGD':
        
            for E in range(self.max_iter):    # epoch

                epoch_loss = 0
                index = np.random.permutation(n)    # shuffle indices for SGD
                
                for i in range(0, n, self.batch_size):
                    X = X_with_bias[index[i:i+self.batch_size]]    # get batch
                    y = y_train.values[index[i:i+self.batch_size]]
                    grad = (2 / self.batch_size) * X.T @ (X @ w - y)  # calculate gradient dL/dw before threshholding for L1
                    
                    w -= self.eta * grad    # update weights

                    w = self._soft_threshold(w, self.alpha * self.eta)  # apply ISTA threshholding for L1
    
                    epoch_loss += np.sum((X @ w - y) ** 2)

                epoch_loss /= n

                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")


        if self.method == 'GD':

            X = X_with_bias
            y = y_train.values
        
            for E in range(self.max_iter):

                grad = 2 * X.T @ (X @ w - y) / n    # gradient using the whole dataset
                    
                w -= self.eta * grad    # update weights

                w = self._soft_threshold(w, self.alpha * self.eta)  # apply ISTA threshholding for L1

                epoch_loss = np.sum((X @ w - y) ** 2) / n
                
                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")

        self.intercept_ = w[0]
        self.coef_ = w[1:]

        return self


    def predict(self, X_test):
        """
        Predict target values for test data.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values
        """
        if self.coef_ is None:
            raise Exception('fit first')

        n = X_test.shape[0]   # n predictions
        if isinstance(X_test, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test])

        w = np.hstack([self.intercept_, self.coef_])      # combine intercept (bias) term with feature coefficients
        
        return X_with_bias @ w



class MyElasticNet():
    """
    Custom ElasticNet Regression implementation with multiple optimization methods.
    
    Supports three training methods:
    - SGD: Stochastic Gradient Descent with mini-batches
    - GD: Full-batch Gradient Descent 
    
    Parameters
    ----------
    method : str, default='SGD'
        Optimization method: 'SGD' or 'GD'
    l1_alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).
    l2_alpha : float, default=1.0
        Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).
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


    Attributes
    ----------
    coef_ : array
        Feature coefficients (weights for input features)
    intercept_ : float
        Bias term (intercept)
    n_iter_ : int
        Actual number of iterations performed during fit
    """
    
    def __init__(self, method='SGD', batch_size=32, max_iter=1000, tol=0.1, eta=0.01, random_state=21, log_freq=100, l1_alpha=1.0, l2_alpha=1.0):
        
        self.method = method
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta  
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

        self.log_freq = log_freq


    def _soft_threshold(self, x, threshold):    # for ISTA
            return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X_train, y_train):
        """
        Fit ElasticNet regression model to training data.
        Iterative Shrinking-Thresholding Algorithm (ISTA) is used for optimization with non-smooth L1 term 
        
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
        n, d = X_train.shape    # n samples, d features 
        w = np.zeros(d + 1)    # initialize weights 
        if isinstance(X_train, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_train])    # add 1 as a first element of sample vectors for operations with weights vector having bias term

        iters_no_change = 0      # to track if loss function does not decrease for a number consecutive epochs (converged)
        self.n_iter_ = self.max_iter

        if self.method == 'SGD':
        
            for E in range(self.max_iter):    # epoch

                epoch_loss = 0
                index = np.random.permutation(n)    # shuffle indices for SGD
                
                for i in range(0, n, self.batch_size):
                    X = X_with_bias[index[i:i+self.batch_size]]    # get batch
                    y = y_train.values[index[i:i+self.batch_size]]
                    grad = (2 / self.batch_size) * X.T @ (X @ w - y) + 2 * self.l2_alpha * w  # calculate gradient dL/dw with L2 before threshholding for L1
                    
                    w -= self.eta * grad    # update weights

                    w = self._soft_threshold(w, self.l1_alpha * self.eta)  # apply ISTA threshholding for L1
    
                    epoch_loss += np.sum((X @ w - y) ** 2)

                epoch_loss /= n

                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")


        if self.method == 'GD':

            X = X_with_bias
            y = y_train.values
        
            for E in range(self.max_iter):

                grad = 2 * X.T @ (X @ w - y) / n + 2 * self.l2_alpha * w  # gradient with L2 term using the whole dataset
                    
                w -= self.eta * grad    # update weights

                w = self._soft_threshold(w, self.l1_alpha * self.eta)  # apply ISTA threshholding for L1

                epoch_loss = np.sum((X @ w - y) ** 2) / n
                
                if E > 0:
                    if np.abs(old_loss - epoch_loss) < self.tol:    # stop criterion if converged
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
    
                    if iters_no_change >= 5:
                        self.n_iter_ = E
                        break
    
                old_loss = epoch_loss

                if E % self.log_freq == 0:      # print logging message
                    print(f"Epoch {E}, Loss: {epoch_loss:.4f}")

        self.intercept_ = w[0]
        self.coef_ = w[1:]

        return self


    def predict(self, X_test):
        """
        Predict target values for test data.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values
        """
        if self.coef_ is None:
            raise Exception('fit first')

        n = X_test.shape[0]   # n predictions
        if isinstance(X_test, pd.DataFrame):
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test.values])
        else:
            X_with_bias = np.hstack([np.ones(n).reshape(n, 1), X_test]) 

        w = np.hstack([self.intercept_, self.coef_])      # combine intercept (bias) term with feature coefficients
        
        return X_with_bias @ w



class MyMinMaxScaler():

    def __init__(self):
        
        self.min_x = None
        self.max_x = None
        self.feature_names_in_ = None


    def fit(self, X, y=None):
        X_copy = X.copy()
        
        self.min_x = X_copy.min()
        self.max_x = X_copy.max()
        self.feature_names_in_ = X.columns
    
        return self
    

    def transform(self, X, y=None):

        X_copy = X.copy()

        return ((X_copy - self.min_x) / (self.max_x - self.min_x))
    

    def fit_transform(self, X, y=None):
        
        return self.fit(X).transform(X) 



class MyStandardScaler():
    """
    pandas built-in .std() and .mean() use N-1 for unbiased estimates,
    original sklearn implementation uses N like numpy.std() and numpy.mean() do
    """
    def __init__(self):
        
        self.mean_x = None
        self.std_x = None
        self.feature_names_in_ = None


    def fit(self, X, y=None):
        X_copy = X.copy()
        
        self.mean_x = np.mean(X_copy, axis=0)
        self.std_x = np.std(X_copy, axis=0)
        self.feature_names_in_ = X.columns
    
        return self
    

    def transform(self, X, y=None):

        X_copy = X.copy()

        return ((X_copy - self.mean_x) / self.std_x)
    

    def fit_transform(self, X, y=None):
        
        return self.fit(X).transform(X) 