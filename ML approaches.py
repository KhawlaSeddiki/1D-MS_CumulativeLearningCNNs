
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:10:49 2020
@author: khawla Seddiki
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn import feature_selection


X = pd.read_csv("../my_spectra1.csv")
y = pd.read_csv("../my_spectra2.csv")


def rf_classifier(X, y, is_default=True):
    from sklearn.ensemble import RandomForestClassifier

    if is_default:
        model = RandomForestClassifier(probability=True)
        model.fit(X, y)
        return model
    else:
        param_grid = {
            'n_estimators': range(10, 100, 10),
            'max_features': np.linspace(0.5, 0.9, num=5).tolist(),
            'max_depth': [10,50,None],
        }
        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
        x_train_fs = fs.fit_transform(X, y)

        model = RandomForestClassifier(probability=True)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',  verbose=1, n_jobs=-1)
        grid_search.fit(x_train_fs, y_train)
        best_parameters = grid_search.best_estimator_.get_params()

        # model with the best parameters
        model = RandomForestClassifier(n_estimators=best_parameters['n_estimators'],
                                       max_features=best_parameters['max_features'],
                                       max_depth=best_parameters['max_depth'], probability=True)
        model.fit(x_train_fs, y_train)
        return model

def svm_classifier(X, y, is_default=True):
    from sklearn.svm import SVC

    if is_default:
        model = SVC(probability=True)
        model.fit(X, y)
        return model
    else:
        param_grid = {
            'kernel': ('rbf'),
            'C': [1e-2, 1e-1, 1, 10],
            'gamma': [1e-4, 1e-3, 1e-2]
        }
        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
        x_train_fs = fs.fit_transform(X, y)

        model = SVC(probability=True)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',  verbose=1, n_jobs=-1)
        grid_search.fit(x_train_fs, y_train)
        best_parameters = grid_search.best_estimator_.get_params()

        # model with the best parameters
        model = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'],
                    gamma=best_parameters['gamma'], probability=True)
        model.fit(x_train_fs, y_train)
        return model

def lda_classifier(X, y, is_default=True):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if is_default:
        model = LinearDiscriminantAnalysis(probability=True)
        model.fit(X, y)
        return model
    else:
        param_grid = {
            'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        pca = PCA(n_components=2)
        #x_train_fs = pca.fit_transform(X)
        model = LinearDiscriminantAnalysis(probability=True)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',  verbose=1, n_jobs=-1)
        grid_search.fit(x_train_fs, y_train)
        best_parameters = grid_search.best_estimator_.get_params()

        # model with the best parameters
        model = LinearDiscriminantAnalysis(n_components=best_parameters['n_components'], probability=True)
        model.fit(x_train_fs, y_train)
        return model



