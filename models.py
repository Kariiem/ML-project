import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from mlxtend.evaluate import bias_variance_decomp
from sklearn.linear_model import RidgeClassifier


# Read the csv file and preprocess it: convert qualitative attributes to integers
train = pd.read_csv("train.csv").drop_duplicates()
X = train.agg(
    utils.transform_dataset
)  # utils.transform_dataset is a dicitionary which applies a transforming function on each column

X["Is_Int"] = 0

X["Is_Int"] = (
    (abs(round(X["Veg_Consump"]) - X["Veg_Consump"]) < 0.01).astype(int)
    + (abs(round(X["Water_Consump"]) - X["Water_Consump"]) < 0.01).astype(int)
    + (abs(round(X["Phys_Act"]) - X["Phys_Act"]) < 0.01).astype(int)
    + (abs(round(X["Time_E_Dev"]) - X["Time_E_Dev"]) < 0.01).astype(int)
    + (abs(round(X["Age"]) - X["Age"]) < 0.01).astype(int)
    + (abs(round(X["Meal_Count"]) - X["Meal_Count"]) < 0.01).astype(int)
)

X["BMI"] = X["Weight"].astype(float) / (X["Height"] ** 2).astype(float)


X_train, X_test, Y_train, Y_test = train_test_split(
    X.drop(columns=["Body_Level"]), X["Body_Level"], test_size=0.2
)

# solve the imbalance problem using SMOTE
sm = SMOTE()
X_train_smote, Y_train_smote = sm.fit_resample(X_train, Y_train)


def svm_model(X_train, Y_train, X_test, Y_test):
    # Train the model
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    # classification report
    y_pred = clf.predict(X_test)
    print("svm: kernel=rbf, C=1.0, gamma='scale'")
    print(classification_report(Y_test, y_pred))
    bvd = bias_variance_decomp(
        clf,
        X_train.values,
        Y_train.values,
        X_test.values,
        Y_test.values,
        random_seed=42,
    )
    print(
        "svm_clf||                Bias: ",
        bvd[0],
        "Variance: ",
        bvd[1],
        "Error: ",
        bvd[2],
        "\n\n",
    )
    return clf


def svm_model_grid_search(X_train, Y_train, X_test, Y_test, parameters):
    # trying grid search on svm
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    print("svm: grid search")
    clf.fit(X_train, Y_train)
    print(clf.best_params_)
    y_pred = clf.predict(X_test)
    print(classification_report(Y_test, y_pred))
    bvd = bias_variance_decomp(
        clf.best_estimator_,
        X_train.values,
        Y_train.values,
        X_test.values,
        Y_test.values,
        random_seed=42,
    )
    print(
        "svm_clf_grid_search||    Bias: ",
        bvd[0],
        "Variance: ",
        bvd[1],
        "Error: ",
        bvd[2],
        "\n\n",
    )
    return clf.best_estimator_


def random_forest_model(X_train, Y_train, X_test, Y_test):
    # trying random forest
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print("random forest: n_estimators=100, max_depth=None")
    print(classification_report(Y_test, y_pred))
    bvd = bias_variance_decomp(
        clf,
        X_train.values,
        Y_train.values,
        X_test.values,
        Y_test.values,
        random_seed=42,
    )
    print(
        "random_forest_clf||      Bias: ",
        bvd[0],
        "Variance: ",
        bvd[1],
        "Error: ",
        bvd[2],
        "\n\n",
    )
    return clf


def naive_bayes_model(X_train, Y_train, X_test, Y_test):
    # trying naive bayes
    clf = naive_bayes.GaussianNB()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print("naive bayes: GaussianNB")
    print(classification_report(Y_test, y_pred))
    bvd = bias_variance_decomp(
        clf,
        X_train.values,
        Y_train.values,
        X_test.values,
        Y_test.values,
        random_seed=42,
    )
    print(
        "naive_bayes_clf||        Bias: ",
        bvd[0],
        "Variance: ",
        bvd[1],
        "Error: ",
        bvd[2],
        "\n\n",
    )
    return clf


def ridge_regression_model_grid_search(X_train, Y_train, X_test, Y_test, parameters):
    # trying ridge regression
    ridge = RidgeClassifier(copy_X=True, random_state=42)
    clf = GridSearchCV(ridge, parameters)
    print("ridge: grid search")
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(classification_report(Y_test, y_pred))
    bvd = bias_variance_decomp(
        clf,
        X_train.values,
        Y_train.values,
        X_test.values,
        Y_test.values,
        random_seed=42,
    )
    print(
        "ridge_regression_clf||   Bias: ",
        bvd[0],
        "Variance: ",
        bvd[1],
        "Error: ",
        bvd[2],
        "\n\n",
    )
    return clf


def svm_model_best_params(X_train, Y_train, X_test, Y_test):
    # Train the model
    clf = svm.SVC(kernel="linear", C=10, gamma=0.01)
    clf.fit(X_train, Y_train)
    # classification report
    y_pred = clf.predict(X_test)
    print("svm: kernel=linear C=10 gamma=0.01")
    print(classification_report(Y_test, y_pred))
    bvd = bias_variance_decomp(
        clf,
        X_train.values,
        Y_train.values,
        X_test.values,
        Y_test.values,
        random_seed=42,
    )
    print(
        "svm_clf||                Bias: ",
        bvd[0],
        "Variance: ",
        bvd[1],
        "Error: ",
        bvd[2],
        "\n\n",
    )
    return clf
