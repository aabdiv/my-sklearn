import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from tqdm.notebook import tqdm
from itertools import product
from functools import reduce

def check_X_y_(X, y):
    """Validate X and y inputs"""

    if X is None or y is None:
        raise ValueError("X and y cannot be None")
    
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(f"X must be a pandas.DataFrame or a NumPy ndarray, got {type(X).__name__}")

    if not isinstance(y, (pd.DataFrame, np.ndarray, pd.Series)):
        raise TypeError(
            f"y must be pandas.DataFrame, pandas.Series, or numpy.ndarray, "
            f"got {type(y).__name__}"
        )

    if X.size == 0 or y.size == 0:
        raise ValueError(f"X and y cannot be empty")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X size does not match y size")

    if len(X.shape) != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

def check_split_input_(
    X,
    y,
    test_size,
    validation_size,
    test_date,
    validation_date,
    date_column
):
    """Validate inputs for data splitting"""

    # validate X and y
    check_X_y_(X, y)

    # for date split
    if date_column is not None or test_date is not None or validation_date is not None:
        if date_column is None or test_date is None:
            raise ValueError(f"Both date_column and test_date must be provided for date-based split")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"date_column requires X to be a pandas.DataFrame")
        
        if date_column not in X.columns:
            raise ValueError(
                f"date_column '{date_column}' not found in X.columns. "
                f"Available columns: {list(X.columns)}"
            )

        # validate date column in data
        if not pd.api.types.is_datetime64_any_dtype(X[date_column]):
            # Try to convert if not datetime
            try:
                _ = pd.to_datetime(X[date_column])
            except Exception as e:
                raise TypeError(
                    f"date_column '{date_column}' must be datetime-like. Error: {e}"
                )
        # validate test_date and validation_date
        try:
            if isinstance(test_date, str):
                test_split_date = pd.to_datetime(test_date)
            elif isinstance(test_date, (pd.datetime, pd.Timestamp)):
                test_split_date = pd.Timestamp(test_date)
            else:
                raise TypeError(
                    f"test_date must be string, datetime, or Timestamp, got {type(test_date).__name__}"
                )
            if validation_date is not None:
                if isinstance(validation_date, str):
                    validation_split_date = pd.to_datetime(validation_date)
                elif isinstance(test_date, (pd.datetime, pd.Timestamp)):
                    validation_split_date = pd.Timestamp(validation_date)
                else:
                    raise TypeError(
                        f"validation_date must be string, datetime, or Timestamp, got {type(validation_date).__name__}"
                    )
        except Exception as e:
            raise ValueError(f"Invalid test_date or validation_date format. Error: {e}")
        
        # check if test_date and validation_date are within data range
        dates = pd.to_datetime(X[date_column])
        min_date = dates.min()
        max_date = dates.max()
        
        if test_split_date < min_date or test_split_date > max_date:
            raise ValueError(
                f"test_date {test_date}must be between data date range "
                f"[{min_date}, {max_date}]"
            )

        if validation_date is not None:
            if validation_split_date < min_date or validation_split_date > max_date:
                raise ValueError(
                    f"validation_date {validation_date} must be between data date range "
                    f"[{min_date}, {max_date}]"
                )
        
            if test_split_date <= validation_split_date:
                raise ValueError(f"test_date must come after validation_date")
            
    # for size-based random split
    else:
        if test_size is None:
            raise ValueError(f"test_size must be provided for size-based split")
        
        if not isinstance(test_size, float):
            raise TypeError(f"test_size must be float, got {type(test_size).__name__}")

        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be a ratio in (0; 1) interval")

        if validation_size is not None:
            if not isinstance(validation_size, float):
                raise TypeError(f"validation_size must be float, got {type(test_size).__name__}")

            if not 0 < validation_size < 1:
                raise ValueError(f"validation_size must be a ratio in [0; 1) half-interval")

            if test_size + validation_size >= 1:
                raise ValueError(f"sum of test size and validation size must be strictly smaller than 1 for test samples to exist")
    
    
def my_data_split(
    X,
    y,
    test_size=0.2,
    validation_size=None,
    test_date=None,
    validation_date=None,
    date_column=None,
    random_state=None
):
    """Split data into train, validation and test sets
    
    Supports two split modes:
    1. Date-based: Provide test_date and date_column (and optionally validation_date)
    2. Size-based: Provide test_size (and optionally validation_size)
    
    Returns:
        If validation not requested: (X_train, X_test, y_train, y_test)
        If validation requested: (X_train, X_valid, X_test, y_train, y_valid, y_test)
    """

    check_split_input_(X, y, test_size, validation_size, test_date, validation_date, date_column)     

    X_copy = X.copy()
    
    if test_date is not None:    # date split
        # convert dates column if needed
        if not pd.api.types.is_datetime64_any_dtype(X[date_column]):
            X_copy[date_column] = pd.to_datetime(X[date_column])
        
        test_split_date = pd.to_datetime(test_date)
        
        if validation_date is not None:
            validation_split_date = pd.to_datetime(validation_date)

            # train, valid, test masks by date
            train_mask = X_copy[date_column] < validation_split_date
            validation_mask = (X_copy[date_column] >= validation_split_date) & (X_copy[date_column] < test_split_date)
            test_mask = X_copy[date_column] >= test_split_date
            
        else:
            # train, test masks by date
            train_mask = X_copy[date_column] < test_split_date
            test_mask = X_copy[date_column] >= test_split_date
            validation_mask = pd.Series(False, index=X_copy.index)  # empty validation
        
        # get indices from masks
        train_indexes = np.where(train_mask)[0]
        validation_indexes = np.where(validation_mask)[0]
        test_indexes = np.where(test_mask)[0]
    else:    # size split
        if random_state:
            np.random.seed(random_state)    # deterministic split
    
        data_n_samples = X_copy.shape[0]    # number of samples
        permuted_indexes = np.random.permutation(data_n_samples)    # permute sample indexes
        
        test_boundary = int(data_n_samples * test_size)     # calculate boundary index for test part
        
        if validation_size is not None:
            validation_boundary = int(data_n_samples * (test_size + validation_size))    # calculate boundary index for validation part
        else:
            validation_boundary = test_boundary
    
        
        test_indexes = permuted_indexes[:test_boundary]
        validation_indexes = permuted_indexes[test_boundary:validation_boundary]
        train_indexes = permuted_indexes[validation_boundary:]
    
    if isinstance(X_copy, pd.DataFrame):
        X_train = X_copy.iloc[train_indexes]
        X_valid = X_copy.iloc[validation_indexes]
        X_test = X_copy.iloc[test_indexes]
        y_train = y.iloc[train_indexes]
        y_valid = y.iloc[validation_indexes]
        y_test = y.iloc[test_indexes]
    
    if isinstance(X_copy, np.ndarray):
        X_train = X_copy[train_indexes]
        X_valid = X_copy[validation_indexes]
        X_test = X_copy[test_indexes]
        y_train = y[train_indexes]
        y_valid = y[validation_indexes]
        y_test = y[test_indexes]
    
    if validation_size is None and validation_date is None:
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_valid, X_test, y_train, y_valid, y_test        
          
        

def check_KFold_input_(X, y, k):
    """Validate inputs for KFold"""

    # validate X and y
    check_X_y_(X, y)

    # validate k
    if k is None:
        raise ValueError(f"k must be provided for KFold")
        
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")

    if k <= 1:
        raise ValueError(f"k must be 2 or larger. got {k=}")
    
    if k > X.shape[0]:
        raise ValueError(f"number of samples cannot be smaller than number of groups. got {X.shape[0]} samples, {k} groups")


def check_GroupedKFold_input_(X, y, k, group_field):
    """Validate inputs for GroupedKFold"""

    # validate X and y
    check_X_y_(X, y)

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas.DataFrame for GroupedKFold, got {type(X).__name__}")

    # validate k
    if k is None:
        raise ValueError(f"k must be provided for KFold")
        
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")

    if k <= 1:
        raise ValueError(f"k must be 2 or larger. got {k=}")
    
    if k > X.shape[0]:
        raise ValueError(f"number of samples cannot be smaller than number of folds. got {X.shape[0]} samples, {k} folds")

    n_of_groups = len(np.unique(X[group_field].values))
    if n_of_groups < k:
        raise ValueError(f"number of groups cannot be smaller than number of folds. got {n_of_groups} groups, {k} folds")

    # validate group_field
    if group_field is None:
        raise ValueError(f"group_field must be provided for GroupedKFold")
    
    if not isinstance(group_field, str):
        raise TypeError(f"group_field must be a string, got {type(group_field).__name__}")
    
    if group_field not in X.columns:
        raise ValueError(
            f"group_field '{group_field}' not found in X.columns. "
            f"Available columns: {list(X.columns)}"
        )


def check_StratifiedKFold_input_(X, y, k, stratify_field):
    """Validate inputs for StratifiedKFold"""

    # validate X and y
    check_X_y_(X, y)

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas.DataFrame for StratifiedKFold, got {type(X).__name__}")

    if not isinstance(y, pd.Series):
        raise TypeError(f"y must be a pandas.Series for StratifiedKFold, got {type(y).__name__}")

    # validate k
    if k is None:
        raise ValueError(f"k must be provided for KFold")
        
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")

    if k <= 1:
        raise ValueError(f"k must be 2 or larger. got {k=}")
    
    if k > X.shape[0]:
        raise ValueError(f"number of samples cannot be smaller than number of folds. got {X.shape[0]} samples, {k} folds")

    # validate stratify_field
    if stratify_field is None:
        raise ValueError(f"stratify_field must be provided for StratifiedKFold")
    
    if not isinstance(stratify_field, str):
        raise TypeError(f"stratify_field must be a string, got {type(stratify_field).__name__}")
    
    if stratify_field not in X.columns and stratify_field != y.name:
        raise ValueError(
            f"stratify_field '{stratify_field}' not found in X.columns and is not y. "
            f"Available columns: {list(X.columns)}, y is {y.name}"
        )

    if stratify_field in X.columns:
        stratify_values = X[stratify_field].values
    elif stratify_field == y.name:
        stratify_values = y.values

    # check if each class in stratify_field has at least k samples
    class_counts = np.unique(stratify_values, return_counts=True)[1]
    if any(k > class_counts):
        raise ValueError(f"each class in stratify_field must have at least k samples. got class counts: {class_counts}")


def check_TimeSeriesSplit_input_(*, X, y, k, date_field=None, by_date_field):
    """Validate inputs for TimeSeriesSplit"""

    # validate X and y
    check_X_y_(X, y)

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas.DataFrame for TimeSeriesSplit, got {type(X).__name__}")

    # validate k
    if k is None:
        raise ValueError(f"k must be provided for TimeSeriesSplit")
        
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")

    if k <= 1:
        raise ValueError(f"k must be 2 or larger. got {k=}")
    
    if k > X.shape[0]:
        raise ValueError(f"number of samples cannot be smaller than number of folds. got {X.shape[0]} samples, {k} folds")

    if by_date_field:
    # validate date_field
        if date_field is None:
            raise ValueError(f"date_field must be provided for TimeSeriesSplit")
    
        if not isinstance(date_field, str):
            raise TypeError(f"date_field must be str, got {type(date_field).__name__}")
    
        if date_field not in X.columns:
            raise ValueError(
                f"date_field '{date_field}' not found in X.columns. "
                f"Available columns: {list(X.columns)}"
            )
    
        if not pd.api.types.is_datetime64_any_dtype(X[date_field]):
                # Try to convert if not datetime
                try:
                    _ = pd.to_datetime(X[date_field])
                except Exception as e:
                    raise TypeError(
                        f"date_field '{date_field}' must be datetime-like. Error: {e}"
                    )
    

def my_KFold(X, y, k=5, shuffle=False, random_state=None):
    """
    K-Fold cross-validation iterator.
    
    Parameters:
        X: Features array or DataFrame
        y: Target array or Series
        k: Number of folds
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
    
    Returns:
        List of tuples (train_indices, test_indices) for each fold
    """
    check_KFold_input_(X, y, k)
    
    X_copy = X.copy()
    data_n_samples = X_copy.shape[0]

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state) 
        indices = np.random.permutation(data_n_samples)
    else:
        indices = np.arange(data_n_samples)

    n_samples_by_fold = np.full(k, fill_value=data_n_samples // k)
    n_samples_by_fold[:data_n_samples % k] += 1     # get number of samples in each of k folds

    test_left_boundary = 0      # sliding window left bound
    test_right_boundary = 0     # sliding window right bound

    folds = np.empty(k, dtype=object)
    
    for i in range(k):
        test_right_boundary += n_samples_by_fold[i]
        
        test_indices = indices[test_left_boundary:test_right_boundary]      # get indicies within the frame 
        train_indices = np.concatenate([indices[0:test_left_boundary], indices[test_right_boundary:]])     # get indicies outside the frame
        
        folds[i] = (train_indices, test_indices)

        test_left_boundary += n_samples_by_fold[i]

    return folds


def my_GroupedKFold(X, y, k=5, group_field=None):
    """
    Grouped K-Fold cross-validation iterator.
    
    Groups are kept together in the same fold (no group is split across folds).
    
    Parameters:
        X: Features DataFrame (must contain group_field column)
        y: Target array or Series
        k: Number of folds
        group_field: Column name containing group identifiers
    
    Returns:
        List of tuples (train_indices, test_indices) for each fold
    """
    check_GroupedKFold_input_(X, y, k, group_field)
    
    X_copy = X.copy()
    data_n_samples = X_copy.shape[0]

    n_samples_by_fold = np.zeros(k)
    groups_by_fold = [[] for _ in range(k)]     # stores group distribution by folds 

    groups, group_counts = np.unique(X[group_field].values, return_counts=True)

    folds = np.empty(k, dtype=object)

    for group_index in group_counts.argsort()[::-1]:    # loop through groups from largest to smallest

        smallest_fold = n_samples_by_fold.argmin()      # get fold with smallest number of samples
        n_samples_by_fold[smallest_fold] += group_counts[group_index]   # increase number of samples in the smallest fold by size of largest group
        groups_by_fold[smallest_fold].append(groups[group_index])       # put name (value) of the group into fold

    for idx, group_list in enumerate(groups_by_fold):

        mask = np.isin(X[group_field].values, group_list)   # get indices of respective samples using created distribution of groups by folds
        test_indices = np.where(mask)[0]
        train_indices = np.where(~mask)[0]
        folds[idx] = (train_indices, test_indices)

    return folds


def my_StratifiedKFold(X, y, k=5, stratify_field=None):
    """
    Stratified K-Fold cross-validation iterator.
    
    Parameters:
        X: Features array or DataFrame
        y: Target array or Series
        k: Number of folds
        stratify_field: Column name to stratify by, or None to use y
    
    Returns:
        List of tuples (train_indices, test_indices) for each fold
    """
    check_StratifiedKFold_input_(X, y, k, stratify_field)
    
    X_copy = X.copy()
    data_n_samples = X_copy.shape[0]
    indices = np.arange(data_n_samples)

    n_samples_by_fold = np.zeros(k)

    if stratify_field in X.columns:
        stratify_values = X[stratify_field].values
    elif stratify_field == y.name:
        stratify_values = y.values 

    classes, class_counts = np.unique(stratify_values, return_counts=True)      # get unqiue classes and their counts from stratify_field

    folds = np.empty(k, dtype=object)
    test_folds = [[] for _ in range(k)]     # indices of each test fold will be stored here

    for class_value in classes:
    
        class_indices = np.where(stratify_values == class_value)[0]     # get all indices of a certain class
        class_n_samples = len(class_indices)
    
        n_samples_by_fold = np.full(k, fill_value=class_n_samples // k)
        n_samples_by_fold[:class_n_samples % k] += 1        # how many samples of this class will go to each resulting test fold 
    
        test_left_boundary = 0
        test_right_boundary = 0     # window will move over indices of this class
        
        for i in range(k):
            
            test_right_boundary += n_samples_by_fold[i]
            
            test_indices = class_indices[test_left_boundary:test_right_boundary]    # get ith test fold share of this class samples 
            
            test_folds[i] = np.concatenate([test_folds[i], test_indices])   # add ith test fold share of this class samples to ith test fold
    
            test_left_boundary += n_samples_by_fold[i]

    test_folds = [test_fold.astype(int) for test_fold in test_folds]    # all test folds are collected here

    for idx, test_fold in enumerate(test_folds):
    
        test_set = set(test_fold)
        train_indices = np.array([int(i) for i in indices if i not in test_set])    # get train indices simply by taking all except those already in test

        test_fold = test_fold.astype(int)
        folds[idx] = (train_indices, test_fold)

    return folds


def my_TimeSeriesSplit(X, y, k=5, date_field=None, by_date_field=True):
    """
    Time Series cross-validator.
    
    Two modes:
    1. Date-based: Split by time intervals (by_date_field=True)
    2. Index-based: Standard expanding window (by_date_field=False)
    
    Parameters:
        X: Features array or DataFrame
        y: Target array or Series
        k: Number of folds
        date_field: Column name containing dates (required for use_dates=True)
        by_date_field: Whether to split by dates or by indices
    
    Returns:
        List of tuples (train_indices, test_indices) for each fold
    """
    check_TimeSeriesSplit_input_(X=X, y=y, k=k, date_field=date_field, by_date_field=by_date_field)
    
    X_copy = X.copy()
    data_n_samples = X_copy.shape[0]
    indices = np.arange(data_n_samples)

    folds = np.empty(k-1, dtype=object)     # only k-1 folds are created in TimeSeriesSplit

    if by_date_field:   # split by datetime-like colum using time intervals
    
        max_date = X_copy[date_field].max()
        min_date = X_copy[date_field].min()
        interval = (max_date - min_date) / k    # time interval for each test fold
    
        train_test_boundary = min_date + interval
        
        for i in range(k-1):
    
            end = train_test_boundary + interval
            train_indices = np.where(X_copy[date_field] <= train_test_boundary)[0]  # get indices using time boundaries 
            test_indices = np.where((X_copy[date_field] > train_test_boundary) & (X_copy[date_field] <= end))[0]
            
            folds[i] = (train_indices, test_indices)
    
            train_test_boundary += interval

    else:   # simple split based on implicit time relations in data

        n_samples_by_fold = np.full(k, fill_value=data_n_samples // k)
        n_samples_by_fold[:data_n_samples % k] += 1

        train_test_boundary = n_samples_by_fold[0]
        
        for i in range(k-1):
            
            end = train_test_boundary + n_samples_by_fold[i+1]
            
            train_indices = indices[:train_test_boundary]
            test_indices = indices[train_test_boundary:end]
            
            folds[i] = (train_indices, test_indices)
    
            train_test_boundary += n_samples_by_fold[i+1]

    return folds


def my_permutation_importance_MAPE(estimator, X, y, n_permutations=10, random_state=None):
    """
    Permutation importance using MAPE scoring.
    
    IMPORTANT: X and y should be VALIDATION/TEST data, not training data!
    
    Parameters:
    -----------
    estimator : fitted estimator
        Must have .predict() method
    X : pandas DataFrame
        Features (validation/test set)
    y : array-like
        Target (validation/test set)
    n_repeats : int, default=10
        Number of permutation repeats
    random_state : int or RandomState, default=None
        Random seed for reproducibility
        
    Returns:
    --------
    dictionary with arrays by the following keys:
        feature, importance_mean, importance_std
    """

    n_features = len(X.columns)
    
    results = {
        'feature': np.ndarray(n_features, dtype=object),
        'importance_mean': np.ndarray(n_features, float),
        'importance_std': np.ndarray(n_features, float)
    }

    if random_state is not None:
        np.random.seed(random_state)

    scorer = mean_absolute_percentage_error
    baseline = scorer(y_pred=estimator.predict(X), y_true=y)    # get baseline metric score

    for idx, column in enumerate(X.columns):

        scores_on_permutations = np.ndarray(n_permutations, dtype=float)    # to store scores after every permutation

        for i in range(n_permutations):
            
            X_permuted = X.copy()
            X_permuted[column] = X_permuted[column].iloc[np.random.permutation(len(X_permuted))].values     # permute feature

            scores_on_permutations[i] = scorer(y_pred=estimator.predict(X_permuted), y_true=y)      # put score in array

        scores_on_permutations = scores_on_permutations - baseline      # calculate importance

        results['feature'][idx] = column
        results['importance_mean'][idx] = scores_on_permutations.mean()
        results['importance_std'][idx] = scores_on_permutations.std()

    return results


def my_GridSearchCV(estimator, param_grid, X, y, scorer, cv):
    """
    Custom GridSearchCV implementation.
    
    Parameters:
    -----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface - .fit() and .predict() methods.
    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values.
    X : array-like
        Training data.
    y : array-like
        Target values.
    scorer : callable
        Function with signature scorer(y_true, y_pred) -> float.
    cv : cross-validation generator or iterable
        Yields (train, test) splits as arrays of indices.

    Returns:
    --------
    cv_results : dict of numpy arrays
        Results dictionary.
    """
    fits_count = reduce(lambda x, y: x * y, map(len, param_grid.values()))     # total count of grid search iterations to make 
    
    param_values = list(param_grid.values())
    types = list(map(type, [el[0] for el in param_values])) + [float, float]    # array of types for all parameters for numpy array in cv_results dict
    
    for i, param_type in enumerate(types):
        if param_type == str:
            types[i] = 'U20'      # for categorical-like string features
            
    param_names = list(param_grid.keys())
    
    cv_results = {}
    
    for param, param_type in zip((param_names + ['score_mean', 'score_std']), types):
        cv_results[param] = np.empty(fits_count, dtype=param_type)        # allocate arrays with respective types for cv_results
    
    combinations = product(*param_values)      # get all possible grid search combinations
    
    for i, comb in tqdm(enumerate(combinations), total=fits_count):     # combinations loop
        
        fit_params = dict(zip(param_names, comb))     # dict of params to pass to estimator
        folds_count = len(cv)

        estimator_with_params = estimator(**fit_params)

        score = np.ndarray(folds_count, dtype=float)

        for k, (train, test) in enumerate(cv):      # cross-validation loop

            y_pred = estimator_with_params.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
            fold_score = scorer(y_pred=y_pred, y_true=y.iloc[test])

            score[k] = fold_score
        
        for param, value in zip(param_names, comb):
            cv_results[param][i] = value
        cv_results['score_mean'][i] = np.mean(score)
        cv_results['score_std'][i] = np.std(score)

    return cv_results


def my_RandomGridSearchCV(estimator, param_grid, X, y, scorer, cv, n_iter=10, random_state=None):
    """
    Custom RandomGridSearchCV implementation.
    
    Parameters:
    -----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface - .fit() and .predict() methods.
    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values.
    X : array-like
        Training data.
    y : array-like
        Target values.
    scorer : callable
        Function with signature scorer(y_true, y_pred) -> float.
    cv : cross-validation generator or iterable
        Yields (train, test) splits as arrays of indices.
    n_iter : int, default=10
        Number of parameter settings that are sampled.
    random_state : int, default=None
        Controls randomness.

    Returns:
    --------
    cv_results : dict of numpy arrays
        Results dictionary.
    """
    combinations_count = reduce(lambda x, y: x * y, map(len, param_grid.values()))     # total count of parameter grid combinations
    
    param_values = list(param_grid.values())
    types = list(map(type, [el[0] for el in param_values])) + [float, float]    # array of types for all parameters for numpy array in cv_results dict
    
    for i, param_type in enumerate(types):
        if param_type == str:
            types[i] = 'U20'      # for categorical-like string features
            
    param_names = list(param_grid.keys())
    
    cv_results = {}
    
    for param, param_type in zip((param_names + ['score_mean', 'score_std']), types):
        cv_results[param] = np.empty(n_iter, dtype=param_type)        # allocate arrays with respective types for cv_results
    
    combinations = list(product(*param_values))      # get all possible grid search combinations

    if random_state is not None:
        np.random.seed(random_state)
        
    combinations_to_try = np.random.choice(combinations_count, size=n_iter, replace=False)     # get n_iter indices randomly
    
    for i, comb_idx in tqdm(enumerate(combinations_to_try), total=n_iter):     # combinations loop

        comb = combinations[comb_idx]     # take combination from list
        
        fit_params = dict(zip(param_names, comb))     # dict of params to pass to estimator
        folds_count = len(cv)

        estimator_with_params = estimator(**fit_params)

        score = np.ndarray(folds_count, dtype=float)

        for k, (train, test) in enumerate(cv):      # cross-validation loop

            y_pred = estimator_with_params.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
            fold_score = scorer(y_pred=y_pred, y_true=y.iloc[test])

            score[k] = fold_score
        
        for param, value in zip(param_names, comb):
            cv_results[param][i] = value
        cv_results['score_mean'][i] = np.mean(score)
        cv_results['score_std'][i] = np.std(score)

    return cv_results