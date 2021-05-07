import pickle
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

from random import *
import random
import torch.nn as nn
import torch
from sklearn.metrics.pairwise import *
from sklearn.metrics import confusion_matrix

from train_predict import bfp_clf_results_cv as bf
def evaluation_metrics(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
    # filter incorrect patch by adjusting threshold
    # y_pred = [1 if p >= 0.005 else 0 for p in y_pred_prob]
    # y_pred = [1 if p >= 0.219 else 0 for p in y_pred_prob]
    print('real positive: {}, real negative: {}'.format(list(y_true).count(1),list(y_true).count(0)))
    print('positive: {}, negative: {}'.format(y_pred.count(1),y_pred.count(0)))
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    return acc, prc, rc, f1, auc_

def bfp_clf_results(train_data, labels, algorithm=None, kfold=5,sample_weight=None):
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    # embedding = np.loadtxt(path)  # be careful with the shape since we don't include the last batch
    # nrows = embedding.shape[0]
    # labels = labels[:nrows]
    # kf = KFold(n_splits=kfold)
    skf = StratifiedKFold(n_splits=kfold,shuffle=True)
    print('Algorithm results:', algorithm)
    accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
    index = [[train, test] for train, test in skf.split(train_data, labels)]

    train_index, test_index = index[0][0], index[0][1]
    x_train, y_train = train_data[train_index], labels[train_index]
    weight = sample_weight[train_index]

    x_test, y_test = train_data[test_index], labels[test_index]
    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000).fit(X=x_train, y=y_train)
    elif algorithm == 'svm':
        clf = SVC(gamma='auto', probability=True, kernel='linear',class_weight='balanced',max_iter=10,
                  tol=0.1).fit(X=x_train, y=y_train)
        # clf = LinearSVC(max_iter=1000).fit(X=x_train, y=y_train)
    elif algorithm == 'nb':
        clf = GaussianNB().fit(X=x_train, y=y_train)
    elif algorithm == 'dt':
        clf = DecisionTreeClassifier().fit(X=x_train, y=y_train,sample_weight=None)
    y_pred = clf.predict_proba(x_test)[:, 1]

    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)
    # accs.append(acc)
    # prcs.append(prc)
    # rcs.append(rc)
    # f1s.append(f1)
    # aucs.append(auc_)
    return acc, prc, rc, f1, auc_

def bfp_clf_results_cv(train_data, labels,test_data,labels_t, algorithm=None,ODS=None):
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    # embedding = np.loadtxt(path)  # be careful with the shape since we don't include the last batch
    # nrows = embedding.shape[0]
    # labels = labels[:nrows]
    # kf = KFold(n_splits=kfold)
    print('Algorithm results:', algorithm)
    accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()

    # the first 130(deduplicated) data is patch-sim dataset that we use for testing, so exclude them.
    x_train, y_train = train_data[130:], labels[130:]

    if ODS == True:
        x_test, y_test = train_data[:139],labels[:139]
    else:
        x_test, y_test = test_data, labels_t

    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression(solver='lbfgs', class_weight={1:1}, max_iter=10000).fit(X=x_train, y=y_train)
    elif algorithm == 'svm':
        clf = SVC(gamma='auto', probability=True, kernel='linear',class_weight='balanced',max_iter=1000,
                  tol=0.1).fit(X=x_train, y=y_train)
        # clf = LinearSVC(max_iter=1000).fit(X=x_train, y=y_train)
    elif algorithm == 'nb':
        clf = GaussianNB().fit(X=x_train, y=y_train)
        # clf = BernoulliNB().fit(X=x_train, y=y_train)
    elif algorithm == 'dt':
        clf = DecisionTreeClassifier().fit(X=x_train, y=y_train,sample_weight=None)
    elif algorithm == 'rf':
        clf = RandomForestClassifier(class_weight={1:1},n_estimators=1000).fit(X=x_train, y=y_train)
    elif algorithm == 'knn':
        clf = KNeighborsClassifier().fit(X=x_train, y=y_train)
    elif algorithm == 'ensemble':
        lr = LogisticRegression(solver='lbfgs',class_weight={1:1}, max_iter=1000).fit(X=x_train, y=y_train)
        rf = RandomForestClassifier().fit(X=x_train, y=y_train)
        knn = KNeighborsClassifier().fit(X=x_train, y=y_train)

    if algorithm == 'ensemble':
        y1 = lr.predict_proba(x_test)[:, 1]
        y2 = rf.predict_proba(x_test)[:, 1]
        y3 = knn.predict_proba(x_test)[:, 1]

        y_pred = np.array([y1, y2, y3])
        y_pred = np.mean(y_pred, axis=0)
    else:
        y_pred = clf.predict_proba(x_test)[:, 1]

    for i in range(1, 10):
        y_pred_tn = [1 if p >= i/10.0 else 0 for p in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
        print('i:{}'.format(i/10),end=' ')
        print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))

    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)
    accs.append(acc)
    prcs.append(prc)
    rcs.append(rc)
    f1s.append(f1)
    aucs.append(auc_)
    print('------------------------------------------------------------------------')
    print('5-fold cross validation')
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (
    np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(), np.array(aucs).mean()))


    import sklearn.metrics as metrics
    # calculate the fpr and tpr for all thresholds of the classification
    preds = y_pred
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # # method I: plt
    # import matplotlib.pyplot as plt
    # plt.rcParams.update({'font.size': 15})
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc,color='black',)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--',color='black',)
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

def get_feature(buggy, patched):
    return subtraction(buggy, patched)

def get_features(buggy, patched):

    subtract = subtraction(buggy, patched)
    multiple = multiplication(buggy, patched)
    cos = cosine_similarity(buggy, patched).reshape((-1,1))
    euc = euclidean_similarity(buggy, patched).reshape((-1,1))

    fe = np.hstack((subtract, multiple, cos, euc))
    return fe

def subtraction(buggy, patched):
    return buggy - patched

def multiplication(buggy, patched):
    return buggy * patched

def cosine_similarity(buggy, patched):
    return paired_cosine_distances(buggy, patched)

def euclidean_similarity(buggy, patched):
    return paired_euclidean_distances(buggy, patched)


if __name__ == '__main__':

    model = 'bert'
    # model = 'doc'
    # model = 'cc2vec_premodel'

    # algorithm
    # algorithm, kfold = 'dt', 5
    algorithm, kfold = 'lr', 5
    # algorithm, kfold = 'nb', 5
    # algorithm, kfold = 'knn', 5
    # algorithm, kfold = 'rf', 5
    # algorithm, kfold = 'ensemble', 5

    # project = 'Chart'
    # project = 'Lang'
    # project = 'Math'
    # project = 'Time'
    project = ''

    # algorithm, kfold = 'svm', 5

    print('model: {}'.format(model))

    path = '../data/experiment3/kui_data_for_' + model + '.pickle'
    # path_test = '../data/experiment3/139_test_data_for_' + model +project+ '.pickle'
    path_test = '../data/experiment3/139_test_data_for_' + model + '.pickle'

    with open(path, 'rb') as input:
    #     while True:
    #         try:
    #             aa = pickle.load(input)
    #             print(aa)
    #         except EOFError:
    #             break
        data = pickle.load(input)
    label, buggy, patched = data
    #
    with open(path_test, 'rb') as input:
    #     while True:
    #         try:
    #             aa = pickle.load(input)
    #             print(aa)
    #         except EOFError:
    #             break
        data = pickle.load(input)
    label_t, buggy_t, patched_t = data

    # data
    train_data = get_features(buggy, patched)
    test_data = get_features(buggy_t, patched_t)
    # train_data = get_feature(buggy, patched)


    # part 1, not using
    # for i in range(1,10):
    # bfp_clf_results_cv(train_data=train_data, labels=label, test_data=None, labels_t=None,algorithm=algorithm,ODS=True)

    # part 2
    bfp_clf_results_cv(train_data=train_data, labels=label, test_data=test_data, labels_t=label_t,algorithm=algorithm,)

