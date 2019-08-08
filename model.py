import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb


########################################################################################################################
# PRE-PROCESSING
########################################################################################################################
def prepare_data(data):
    """
    get rid of redundant feature
    split data into features and labels
    """
    drop_list = ['address', 'attributes', 'business_id', 'categories', 'city', 'hours',
                 'latitude', 'longitude', 'name', 'postal_code', 'state', 'stars', ]

    data = data.drop(columns=drop_list).dropna()
    data = data.replace([np.inf, -np.inf], 0)
    X = data.drop(columns='is_open')
    y = data['is_open']

    return X, y


def resampling_data(data):
    """
    oversampling closed business data
    retrain test data before sampling
    """
    # separate features and labels
    X, y = prepare_data(data)
    # split the data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # concatenate training data back
    t_data = pd.concat([X_train, y_train], axis=1)
    # separate open and close businesses
    data_open = t_data[t_data['is_open'] == 1]
    data_close = t_data[t_data['is_open'] == 0]
    # oversampling minority
    n_samples = int(len(data_open) * 0.8)
    data_close_sampled = resample(data_close, replace=True, n_samples=n_samples, random_state=42)
    # combine majority and oversampled minority
    data_sampled = pd.concat([data_open, data_close_sampled]).sample(frac=1)

    return data_sampled, X_test, y_test


########################################################################################################################
# TUNING PARAMETERS
########################################################################################################################
def parameter_tuning(data):
    """
    tune selected parameters with RandomizedSearchCV
    """
    print('Tuning parameters ...')
    X_train = data.drop(columns='is_open')
    y_train = data['is_open']
    parameters = {
        'num_leaves': [10, 30, 50, 80],
        'min_data_in_leaf': [100, 300, 700, 1000],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
    }

    # apply GridSearch on random forest regression to find the best parameter
    rs_lgb = RandomizedSearchCV(lgb.LGBMClassifier(objective='binary', n_estimators=1000),
                                parameters,
                                cv=5,
                                scoring='recall',  # 'f1'
                                n_jobs=-1,
                                verbose=2,
                                n_iter=20)

    rs_lgb.fit(X_train, y_train)
    print('The best parameters are:\n')
    print(rs_lgb.best_estimator_)

    mean = rs_lgb.cv_results_['mean_test_score']
    std = rs_lgb.cv_results_['std_test_score']
    for mean, std, params in zip(mean, std, rs_lgb.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    return rs_lgb.best_estimator_


########################################################################################################################
# MODEL TRAINING
########################################################################################################################
def lightgbm_model(data_train, X_test, y_test, paras=False):
    # seprate features and labels
    X_train = data_train.drop(columns='is_open')
    y_train = data_train['is_open']

    # assign tuned parameters as default
    if paras:
        parameters = paras
    else:
        parameters = {
            "n_estimators": 1000,
            "objective": 'binary',
            "num_leaves": 50,
            "min_data_in_leaf": 300,
            "learning_rate": 0.2,
            "n_jobs": -1,
        }

    # initialize lightGBM model
    lgb_model = lgb.LGBMClassifier(**parameters)
    # fit the model
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    # results
    print(classification_report(y_test, y_pred))
    print('-----------------------------------------------------------')
    # get confusion matrix
    print('Confusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('-----------------------------------------------------------')
    accuracy = str((cm[0][0] + cm[1][1])/np.sum(cm) * 100)[:5]+'%'
    print("Accuracy is: {}".format(accuracy))


########################################################################################################################
def main(data, tune_parameter=False):
    """
    execute model training
    """
    data_sampled, X_test, y_test = resampling_data(data)
    if tune_parameter:
        best_paras = parameter_tuning(data_sampled)
        lightgbm_model(data_sampled, X_test, y_test, best_paras.get_params())
    else:
        lightgbm_model(data_sampled, X_test, y_test)

