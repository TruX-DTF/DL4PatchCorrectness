import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from subprocess import *
from bert_serving.client import BertClient
from gensim.models import Doc2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import *
import re
import json
import pickle
import numpy as np


def learned_feature(patch_diff, w2v):
    try:
        bugy_all = get_diff_files_frag(patch_diff, type='buggy')
        patched_all = get_diff_files_frag(patch_diff, type='patched')
    except Exception as e:
        print('name: {}, exception: {}'.format(patch_diff, e))
        return []

    # tokenize word
    bugy_all_token = word_tokenize(bugy_all)
    patched_all_token = word_tokenize(patched_all)

    try:
        bug_vec, patched_vec = output_vec(w2v, bugy_all_token, patched_all_token)
    except Exception as e:
        print('name: {}, exception: {}'.format(patch_diff, e))
        return []

    bug_vec = bug_vec.reshape((1, -1))
    patched_vec = patched_vec.reshape((1, -1))

    # embedding feature cross
    subtract, multiple, cos, euc = multi_diff_features(bug_vec, patched_vec)
    embedding = np.hstack((subtract, multiple, cos, euc,))

    # embedding = subtraction(bug_vec, patched_vec)

    return list(embedding.flatten()), bugy_all+patched_all

def subtraction(buggy, patched):
    return patched - buggy

def multiplication(buggy, patched):
    return buggy * patched

def cosine_similarity(buggy, patched):
    return paired_cosine_distances(buggy, patched)

def euclidean_similarity(buggy, patched):
    return paired_euclidean_distances(buggy, patched)

def multi_diff_features(buggy, patched):
    subtract = subtraction(buggy, patched)
    multiple = multiplication(buggy, patched)
    cos = cosine_similarity(buggy, patched).reshape((1, 1))
    euc = euclidean_similarity(buggy, patched).reshape((1, 1))

    return subtract, multiple, cos, euc

def output_vec(w2v, bugy_all_token, patched_all_token):

    if w2v == 'bert':
        m = BertClient(check_length=False, check_version=False,)
        bug_vec = m.encode([bugy_all_token], is_tokenized=True)
        patched_vec = m.encode([patched_all_token], is_tokenized=True)
    elif w2v == 'doc':
        # m = Doc2Vec.load('../model/doc_file_64d.model')
        m = Doc2Vec.load('../model/Doc_frag_ASE.model')
        bug_vec = m.infer_vector(bugy_all_token, alpha=0.025, steps=300)
        patched_vec = m.infer_vector(patched_all_token, alpha=0.025, steps=300)
    else:
        print('wrong model')
        raise

    return bug_vec, patched_vec

def get_diff_files_frag(patch_diff, type):
    # with open(path_patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        flag = True
        # try:
        for line in patch_diff:
            line = line.strip()
            if '*/' in line:
                flag = True
                continue
            if flag == False:
                continue
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                if line.startswith('Index') or line.startswith('==='):
                    continue
                elif '/*' in line:
                    flag = False
                    continue
                elif type == 'buggy':
                    if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                        continue
                        # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        # lines += ' '.join(line) + ' '
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('+'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                elif type == 'patched':
                    if line.startswith('+++'):
                        continue
                        # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        # lines += ' '.join(line) + ' '
                    elif line.startswith('+'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('-'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
        # except Exception:
        #     print(Exception)
        #     return 'Error'
        return lines