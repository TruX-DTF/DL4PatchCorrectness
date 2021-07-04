# from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec, Doc2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


# path_patch_train_all = ['../data/train_data5_frag.txt', '../data/test_data_frag_all_for_doc.txt','../data/kui_data_frag_all_for_doc.txt']
path_patch_train = ['../data/experiment1/train_data5_frag.txt']

def load_data(data_path, bugName=None):

    # bugName to be used to select a specific bug
    # data = np.loadtxt(data_path, dtype=str,comments=None, delimiter='<ml>')
    result = []
    for file in data_path:
        result.append(pd.read_csv(file, header=None, sep='<ml>'))

    df = pd.concat(result)
    # df = pd.read_csv(data_path,sep='<ml>')
    df.columns = ["label", "bugid", "buggy", "patched"]
    # df = pd.DataFrame(data,dtype=str,columns=['label','bugid','buggy','patched'])
    #bugname experiment
    if bugName != None:
        df = df.loc[df['bugid'].str.startswith(bugName)]
    print('the number of train patches is {}'.format(df.shape[0]))
    return df

def Doc(df):
    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
    df['patched'] = df['patched'].map(lambda x: word_tokenize(x))

    data = list(df['buggy']) + list(df['patched'])
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
    model = Doc2Vec(documents, vector_size=128, window=5, min_count=1, workers=4)
    # model.save('../data/doc_frag_all.model')
    model.save('../data/doc_frag.model')

if __name__ == '__main__':
    # df = load_data(data_path,'patch_quicksort')
    # df = load_data(data_path_whole)

    model = 'doc'
    print('train model:{}'.format(model))

    df = load_data(path_patch_train)
    Doc(df)