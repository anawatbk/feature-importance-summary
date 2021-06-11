import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, r2_score
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, load_boston
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import matplotlib.pyplot as plt

def spearmanr_importance(X, y):
    dataset = np.append(X, y.reshape(-1,1), axis=1)
    correlation_matrix = spearmanr(dataset)
    feature_importance = correlation_matrix.correlation[:, -1][:-1]
    return np.abs(feature_importance)


def mRMR(X, y):
    S_size = X.shape[1]-1
    spearmanr_imp = spearmanr_importance(X, y)
    sum_spearmanr = spearmanr(X).correlation
    sum_other_features = (np.sum(np.abs(sum_spearmanr), axis=1) - 1) / S_size
    return spearmanr_imp - sum_other_features



def permutation_importances(model, X, y, metric='accuracy', random_state=99, shuffle_y=False) -> list:
    '''
    Parameters
    ----------
    X : np.array
    y : np.array
    metric : str, default='accuracy'
    
    Returns
    ----------
    feature_imprtance : list
    '''
    if shuffle_y:
        y = np.random.permutation(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model.fit(X_train, y_train)
    # calculate baseline accuracy
    if metric == 'accuracy':
        baseline_score = accuracy_score(y_test, model.predict(X_test))
    else:
        baseline_score = r2_score(y_test, model.predict(X_test))
    # calculate feature importance score following the order of X columns
    feature_importance = []
    for i in range(X.shape[1]):
        original_col = X_test[:, i].copy()
        X_test[:, i] = np.random.permutation(X_test[:, i]) # permute a column of X
        if metric == 'accuracy': 
            permutation_score = accuracy_score(y_test, model.predict(X_test)) # compute from validation set
        else:
            permutation_score = r2_score(y_test, model.predict(X_test)) # compute from validation set
        feature_importance.append(baseline_score - permutation_score)
        X_test[:, i] = original_col # restore original column
        
    return feature_importance


def dropcol_importances(model, X, y, metric='accuracy', random_state=99) -> list:
    '''
    Parameters
    ----------
    X : np.array
    y : np.array
    metric : str, default='accuracy'
    
    Returns
    ----------
    feature_imprtance : list
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model_ = clone(model)
    model.fit(X_train, y_train)
    # calculate baseline accuracy
    if metric == 'accuracy':
        baseline_score = accuracy_score(y_test, model.predict(X_test))
    else:
        baseline_score = r2_score(y_test, model.predict(X_test))
    # calculate feature importance score following the order of X columns
    feature_importance = []
    for i in range(X.shape[1]):
        drop_mask = np.array([True] * X_train.shape[1])
        drop_mask[i] = False
        model_ = clone(model)
        model_.fit(X_train[:, drop_mask], y_train)
        if metric == 'accuracy':
            drop_score = accuracy_score(y_test, model_.predict(X_test[:, drop_mask])) # compute from validation set
        else:
            drop_score = r2_score(y_test, model_.predict(X_test[:, drop_mask])) # compute from validation set
        feature_importance.append(baseline_score - drop_score)
        
    return feature_importance


def feature_importance_std(model, X, y, method='permutation', metric='accuracy', iteration=30, random_state=99):
    '''
    Returns
    ----------
    (feature_importance, feature_importance_std) : tuple
        mean and std of feature_importance
    '''
    n = X.shape[0]
    num_features = X.shape[1]
    variance = []
    for i in range(iteration):
        bootstrap_idxs = resample(np.arange(n), replace=True, n_samples=n) # get bootstrapped indexes
        X_bootstrapped, y_bootstrapped = X[bootstrap_idxs], y[bootstrap_idxs]
        if method == 'permutation':
            result = permutation_importances(model, X_bootstrapped, 
                                             y_bootstrapped, metric=metric, random_state=random_state)
        variance.append(result)
    return np.mean(np.array(variance), axis=0), np.std(np.array(variance), axis=0) / np.sqrt(n)


def plot_feature_importances(feature_importances, feature_names, std=None):
    '''
    Parameters
    ----------
    feature_importances : list
    feature_names : list
    
    Returns
    ----------
    feature_names : list
        list of best k features
    '''
    text_grey = '#6D6766'
    dark_grey = '#010101'
    blue = '#5491F7'
    summary_df = pd.DataFrame()
    summary_df['feature_names'] = feature_names
    summary_df['feature_importances'] = feature_importances
    summary_df.sort_values(by='feature_importances', ascending=True, inplace=True)
    # plots
    fig, ax = plt.subplots(figsize=(10,6))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines['bottom'].set_color(text_grey)
    ax.tick_params(axis='y', labelsize=14, labelcolor=dark_grey, width=0)
    ax.tick_params(axis='x', labelsize=14, labelcolor=text_grey, width=1)
    ax.barh(summary_df['feature_names'], summary_df['feature_importances'], color=blue, height=0.6)
    if std is not None:
        ax.errorbar(summary_df['feature_importances'], summary_df['feature_names'], 
                    xerr=2*std, fmt=',', ecolor=text_grey, mfc=text_grey, capsize=4, markeredgewidth=2)
    ax.set_title('Feature Importances', fontsize=20, color=dark_grey)
    plt.show()


def automatic_feature_selection(model, X, y, feature_importances:list,
                                feature_names:list, metric='accuracy', tol=0.002, random_state=99):
    '''
    Backward Elimination stop the elimination process if score > tol
    
    Returns
    ----------
    None
        plot feature importances
    '''
    num_features = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model.fit(X_train, y_train)
    if metric == 'accuracy':
        baseline_score = accuracy_score(y_test, model.predict(X_test))
    else:
        baseline_score = r2_score(y_test, model.predict(X_test))
    # setup sorted feature importances
    summary_df = pd.DataFrame()
    summary_df['feature_names'] = feature_names
    summary_df['feature_importances'] = feature_importances
    summary_df = summary_df.reset_index().sort_values(by='feature_importances', ascending=False)
    sorted_idx = list(summary_df['index'])
    drop_mask = np.array([True] * num_features)
    for i in range(num_features):
        drop_idx = sorted_idx.pop() # drop lowest importances
        drop_mask[drop_idx] = False
        model_ = clone(model)
        model_.fit(X_train[:, drop_mask], y_train)
        if metric == 'accuracy':
            drop_score = accuracy_score(y_test, model_.predict(X_test[:, drop_mask])) # compute from validation set
        else:
            drop_score = r2_score(y_test, model_.predict(X_test[:, drop_mask])) # compute from validation set
        score = baseline_score - drop_score
        if score > tol:
            drop_mask[drop_idx] = True
            return list(np.array(feature_names)[drop_mask])
        
    return list(np.array(feature_names)[drop_mask])


def plot_empirical_p_value(real_importance, null_dist, feature_names, show_all=False, i=0):
    '''
    Parameters
    ----------
    real_importance : list
    null_dist : list of list
    feature_names: list
    show_all: bool
    i : int
        index of feature_names
    
    Returns
    ----------
    None
        plot nulldist vs target
    '''
    text_grey = '#6D6766'
    dark_grey = '#010101'
    blue = '#5491F7'
    if show_all:
        fig, axes = plt.subplots(ncols=3, nrows=int(np.ceil(len(feature_names) / 3)), figsize=(20,20))
        axes = axes.flatten()
        for i in range(len(feature_names)):
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["top"].set_visible(False)
            axes[i].tick_params(axis='y', labelsize=14, labelcolor=dark_grey, width=0)
            axes[i].tick_params(axis='x', labelsize=14, labelcolor=dark_grey, width=1)
            a = axes[i].hist(null_dist[:, i], label=f'{feature_names[i]} Null importance')
            axes[i].vlines(x=real_importance[i], ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
            axes[i].set_title(f'Importance Distribution of {feature_names[i]}', fontsize=16, color=dark_grey)
            axes[i].legend()
    else:
        real_higher_count = sum(real_importance[i] > null_dist[:, i])
        n = len(null_dist[:, i])
        empirical_p = 1 - (real_higher_count/n)
        print(f'Hypothesis Test (p_threshold=0.05)\n------------------------------')

        if empirical_p < 0.05:
            print(f'empirical p-value = {empirical_p:.3f} < 0.05')
            print(f'Summary: {feature_names[i]} Importance is significant\n\n')
        else:
            print(f'empirical p-value = {empirical_p:.3f} > 0.05')
            print(f'Summary: {feature_names[i]} Importance is not significant\n\n')    

        # plots

        fig, ax = plt.subplots(figsize=(10,6))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='y', labelsize=14, labelcolor=dark_grey, width=0)
        ax.tick_params(axis='x', labelsize=14, labelcolor=dark_grey, width=1)
        a = ax.hist(null_dist[:, i], label=f'{feature_names[i]} Null importance')
        ax.vlines(x=real_importance[i], ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
        ax.set_title(f'Importance Distribution of {feature_names[i]}', fontsize=16, color=dark_grey)
        ax.legend()
    plt.tight_layout()
    plt.show()


def get_k_scores(model, X, y, k, feature_importance, feature_names, random_state=41):
    num_features = X.shape[1]
    summary_df = pd.DataFrame()
    summary_df['feature_names'] = feature_names
    summary_df['feature_importances'] = feature_importance
    summary_df = summary_df.reset_index().sort_values(by='feature_importances', ascending=False)
    sorted_idx = list(summary_df['index'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    num_features = X.shape[1]
    r2_scores = []
    mask = np.array([False] * num_features)
    for i in range(k):
        train_idx = sorted_idx.pop(0) # drop lowest importances
        mask[train_idx] = True
        model.fit(X_train[:, mask], y_train)
        score = r2_score(y_test, model.predict(X_test[:, mask])) # compute from validation set
        r2_scores.append(score)
    return r2_scores