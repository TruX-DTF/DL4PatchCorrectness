from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec,Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


data_path = '../data/pre_data_new.txt'
data_path_whole = '../data/pre_data_whole.txt'
data_path_kui = '../data/pre_data_kui.txt'

path_patch_test = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_test'
path_patch_train = '../data/train_data5.txt'

def load_data(data_path, bugName=None):

    # bugName to be used to select a specific bug
    data = np.loadtxt(data_path, dtype=str,comments=None, delimiter='<ml>')
    if len(data) == 4:
        data = np.array([data])
    df = pd.DataFrame(data,dtype=str,columns=['label','bugid','buggy','patched'])

    #bugname experiment
    if bugName != None:
        df = df.loc[df['bugid'].str.startswith(bugName)]
    print('the number of train patches is {}'.format(df.shape[0]))
    return df

def core(df,model):
    if model == 'bert':
        bert(df)
    elif model == 'Doc_whole':
        Doc_whole(df)
    elif model == 'doc':
        Doc(df)
    else:
        print('wrong model')

def bert(df):
    len = df.shape[0]
    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
    df['patched'] = df['patched'].map(lambda x: word_tokenize(x))
    # result = cosine_similarity(bc.encode(list(np.array(df_quick['buggy']))),bc.encode(list(np.array(df_quick['patched']))))
    df['simi'] = None
    bc = BertClient(check_length=False)
    for index,row in df.iterrows():
        print('{}/{}'.format(index,len))
        if row['buggy'] == [] or row['patched'] == []:
            print('buggy or patched is []')
            continue
        try:
            bug_vec = bc.encode([row['buggy']],is_tokenized=True)
            patched_vec = bc.encode([row['patched']],is_tokenized=True)
        except Exception as e:
            print(e)
            continue
        result = cosine_similarity(bug_vec,patched_vec)
        df.loc[index,'simi'] = float(result[0][0])
    df = df.sort_values(by='simi')
    print('the minimum similarity is {}'.format(df['simi'].head(1).values[0]))
    print('the average similarity is {}'.format(np.mean(np.array(df[['simi']]))))
    print('the median similarity is {}'.format(np.median(np.array(df[['simi']]))))

    re = ''
    re += 'the minimum similarity is {}'.format(df['simi'].head(1).values[0]) + '\n'
    re += 'the average similarity is {}'.format(np.mean(np.array(df[['simi']]))) + '\n'
    re += 'the median similarity is {}'.format(np.median(np.array(df[['simi']]))) + '\n'

    np.savetxt(r'../data/train_result_doc.txt', df[['bugid', 'simi']].values, fmt='%s', header=re)
    # df[['bugid','simi']].to_csv('../data/train_result_bert.txt', header=None, index=None, sep=' ', mode='a+')


def Doc_whole(df):
    bug = df['buggy'][0]
    data = list(df['patched'])
    data.append(bug)
    documents = [TaggedDocument([doc], [i]) for i, doc in enumerate(data)]
    model = Doc2Vec(documents, vector_size=64, window=5, min_count=1, workers=4)
    # infered_vec = model.infer_vector(bug,alpha=0.025,steps=100)
    result = model.most_similar_cosmul(positive=bug,topn=5)
    for r in result:
        doc = r[0]
        sim = r[1]
        print(df[df['patched']==doc]['bugid'].values[0],sim)
    pass

def Doc(df):
    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
    df['patched'] = df['patched'].map(lambda x: word_tokenize(x))
    df['simi'] = None

    data = list(df['buggy']) + list(df['patched'])
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
    model = Doc2Vec(documents, vector_size=64, window=5, min_count=1, workers=4)

    for index, row in df.iterrows():
        bug_vec = model.infer_vector(row['buggy'],alpha=0.025,steps=300)
        patched_vec = model.infer_vector(row['patched'],alpha=0.025,steps=300)
        # similarity calculation
        result = cosine_similarity(bug_vec.reshape((1,-1)), patched_vec.reshape((1,-1)))
        df.loc[index, 'simi'] = result[0][0]
    df = df.sort_values(by='simi')
    print('the minimum similarity is {}'.format(df['simi'].head(1).values[0]))
    print('the average similarity is {}'.format(np.mean(np.array(df[['simi']]))))
    print('the median similarity is {}'.format(np.median(np.array(df[['simi']]))))

    re = ''
    re += 'the minimum similarity is {}'.format(df['simi'].head(1).values[0]) + '\n'
    re += 'the average similarity is {}'.format(np.mean(np.array(df[['simi']]))) + '\n'
    re += 'the median similarity is {}'.format(np.median(np.array(df[['simi']]))) + '\n'

    np.savetxt(r'../data/train_result_doc.txt', df[['bugid','simi']].values, fmt='%s', header=re)
    # df[['bugid','simi']].to_csv('../data/train_result_doc.txt', header=None, index=None, sep=' ', mode='a')
    # with open('../data/train_result_doc.txt','w+') as f:
        # f.write(re)
if __name__ == '__main__':
    # df = load_data(data_path,'patch_quicksort')
    # df = load_data(data_path_whole)

    # model = 'bert'
    model = 'doc'
    df = load_data(path_patch_train)
    core(df, model=model)