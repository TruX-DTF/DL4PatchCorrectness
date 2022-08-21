import sys
import os.path
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

from sklearn.metrics.pairwise import *
from joblib import dump, load
import word2vec


def evaluation_metrics(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
    # filter incorrect patch by adjusting threshold
    # y_pred = [1 if p >= 0.039 else 0 for p in y_pred_prob]
    # print('real positive: {}, real negative: {}'.format(list(y_true).count(1),list(y_true).count(0)))
    # print('positive: {}, negative: {}'.format(y_pred.count(1),y_pred.count(0)))
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
        clf = LogisticRegression(solver='lbfgs', max_iter=1000,tol=0.2).fit(X=x_train, y=y_train)
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

def train_model(train_data, labels, algorithm=None,):
    print('training ... ')
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    print('Algorithm:', algorithm)
    x_train, y_train = train_data, labels
    # weight = sample_weight[train_index]

    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)
    elif algorithm == 'svm':
        clf = SVC(gamma='auto', probability=True, kernel='linear',class_weight='balanced',max_iter=1000,
                  tol=0.1).fit(X=x_train, y=y_train)
        # clf = LinearSVC(max_iter=1000).fit(X=x_train, y=y_train)
    elif algorithm == 'nb':
        clf = GaussianNB().fit(X=x_train, y=y_train)
    elif algorithm == 'dt':
        clf = DecisionTreeClassifier().fit(X=x_train, y=y_train,sample_weight=None)

    dump(clf, '../data/model/{}_bert.joblib'.format(algorithm))
    return


def learn_embedding(embedding_method, patch_txt):
    w = Word2vector(embedding_method)
    try:
        patch_vector = w.embedding(patch_txt)
    except Exception as e:
        print(e)
        raise e
    return patch_vector


def predict(x_test, algorithm):
    threshold = 0.5
    print('loading model ... ')
    clf = load('../data/model/{}_bert.joblib'.format(algorithm))

    x_test = np.array(x_test).reshape((1,-1))
    # y_pred_prob = clf.predict_proba(x_test)[:, 1]
    y_pred_prob = clf.predict_proba(x_test)[0,1]
    y_pred = 1 if y_pred_prob >= threshold else 0

    print('############################################')
    if y_pred == 1:
        print('Congrats! Your patch is CORRECT.')
    elif y_pred == 0:
        print('Sorry, your patch is INCORRECT.')


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
    if len(sys.argv) == 3:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]

        if arg1 != 'predict':
            raise 'sorry, we only accept "predict" as task.'
    else:
        raise 'sorry, we require 2 parameters.'


    model = 'bert'
    # model = 'doc'
    # model = 'cc2vec_premodel'

    # algorithm
    # algorithm, kfold = 'dt', 5
    algorithm, kfold = 'lr', 5
    # algorithm, kfold = 'nb', 5

    # algorithm, kfold = 'svm', 5

    print('model: {}'.format(model))

    path = '../data/experiment3/kui_data_for_' + model + '.pickle'
    # path_test = '../data/experiment3/139_test_data_for_' + model + '.pickle'

    with open(path, 'rb') as input:
        data = pickle.load(input)
    label, buggy, patched = data

    index_p = list(np.where(label == 1)[0])
    index_n = list(np.where(label == 0)[0])

    # print('the number of dataset: {}'.format(len(label)))
    # print('positive: {}, negative: {}'.format(len(index_p), (len(label)-len(index_p))))

    # data
    train_data = get_features(buggy, patched)
    # train_data = get_feature(buggy, patched)

    # train_data = train_data[index_all]
    # label = label[index_all]

    # train
    if not os.path.exists('../data/model/{}_bert.joblib'.format(algorithm)):
        train_model(train_data=train_data, labels=label, algorithm=algorithm, )

    patch_txt = arg2
    x_test_vector, _ = word2vec.learned_feature(patch_txt, 'bert')
    predict(x_test_vector, algorithm)

