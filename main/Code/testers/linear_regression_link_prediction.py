import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, average_precision_score, auc, f1_score


def run_linear_regression(path, aa_only):
    data = np.loadtxt(path, delimiter=',')

    if aa_only:
        data_x = data[:, :1]
    else:
        data_x = data[:, :2]

    # poly = PolynomialFeatures(1)
    # data_x = poly.fit_transform(data_x)
    data_y = data[:, -1]

    coefs = []
    aurocs = []
    avg_ps = []

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(data_x):
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        if sum(y_train) < 1 or sum(y_test) < 1:
            continue

        # regr = linear_model.LinearRegression()
        regr = linear_model.LogisticRegression(max_iter=1000, C=1)
        regr.fit(X_train, y_train)
        coefs.append(regr.coef_[0])
        
        # Prediction Scores
        y_score = regr.predict_proba(X_test)[:, -1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
        auroc = auc(fpr, tpr)
        average_precision = average_precision_score(y_test, y_score)

        # f1 = f1_score(y_test, y_score, average='binary')
        # f1s.append(f1)

        aurocs.append(auroc)
        avg_ps.append(average_precision)

    coefs = np.array(coefs)
    # print("Mean AA Coefficient: {0}".format(np.mean(coefs[:, 0])))
    # print("Mean DCAA Coefficient: {0}".format(np.mean(coefs[:, 1])))
    # print("Mean AUROC: {0}".format(np.mean(aurocs)))
    # print("Mean Avg Precision: {0}".format(np.mean(avg_ps)))

    if aa_only:
        return np.mean(coefs[:, 0]), np.mean(aurocs), np.mean(avg_ps)
    else:
        return np.mean(coefs[:, 0]), np.mean(coefs[:, 1]), np.mean(aurocs), np.mean(avg_ps)


