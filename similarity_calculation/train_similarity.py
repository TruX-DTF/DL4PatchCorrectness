from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec, Doc2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from utils import my_split
import numpy as geek

data_path = '../data/pre_data_new.txt'
data_path_whole = '../data/pre_data_whole.txt'
data_path_kui = '../data/pre_data_kui.txt'

# path_patch_test = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_test'
path_patch_train = '../data/experiment1/train_data5_frag.txt'
path_patch_train_incorrect = '../data/experiment1/train_data5_frag_incorrect.txt'

code2vec_path = '../pretrained_models/code2vec_token.txt'

def load_data(data_path, bugName=None):

    # bugName to be used to select a specific bug
    # data = np.loadtxt(data_path, dtype=str,comments=None, delimiter='<ml>')
    # df = pd.DataFrame(data,dtype=str,columns=['label','bugid','buggy','patched'])

    df = pd.read_csv(data_path, sep='<ml>', header=None)
    df.columns = ["label", "bugid", "buggy", "patched"]

    #bugname experiment
    if bugName != None:
        df = df.loc[df['bugid'].str.startswith(bugName)]
    print('the number of train patches is {}'.format(df.shape[0]))
    return df

def core(df,model,supply=None):
    if model == 'bert':
        bc = BertClient(check_length=False)
        bert(df,bc,supply)
    elif model == 'Doc_whole':
        Doc_whole(df)
    elif model == 'doc':
        # m = Doc2Vec.load('../data/doc_frag.model')
        Doc(df,supply,m=None)
    elif model == 'code2vec':
        code2vec(df, code2vec_path)
    else:
        print('wrong model')

def bert(df,bc,supply):
    length = df.shape[0]
    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
    df['patched'] = df['patched'].map(lambda x: word_tokenize(x))

    # result = cosine_similarity(bc.encode(list(np.array(df_quick['buggy']))),bc.encode(list(np.array(df_quick['patched']))))
    df['simi'] = None

    # df.iloc[:, 'buggy'] = bc.encode(list(df['buggy']), is_tokenized=True)
    # df.iloc[:, 'patched'] = bc.encode(list(df['patched']), is_tokenized=True)

    # tmp
    # for index,row in df.iterrows():
    #     print('{}/{}'.format(index,length))
    #     if row['buggy'] == [] or row['patched'] == []:
    #         print('buggy or patched is []')
    #         continue
    #     try:
    #         df.loc[index, ['buggy']] = str(list(bc.encode([row['buggy']],is_tokenized=True)[0]))
    #         df.loc[index, ['patched']] = str(list(bc.encode([row['patched']],is_tokenized=True)[0]))
    #     except Exception as e:
    #         print(e)
    #         continue
    # result = cosine_similarity(list(df['buggy']),list(df['patched']))

    for index,row in df.iterrows():
        print('{}/{}'.format(index,length))
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
    # re += 'the minimum similarity is {}'.format(df['simi'].head(1).values[0]) + '\n'
    # re += 'the average similarity is {}'.format(np.mean(np.array(df[['simi']]))) + '\n'
    # re += 'the median similarity is {}'.format(np.median(np.array(df[['simi']]))) + '\n'

    # np.savetxt(r'../data/train_result_bert.txt', df[['bugid', 'simi']].values, fmt='%s', header=re)
    if supply == True:
        df[['bugid','simi']].to_csv('../data/experiment1/train_result_frag_bert_incorrect.csv', header=None, index=None, sep=' ', mode='w')
    else:
        df[['bugid','simi']].to_csv('../data/experiment1/train_result_frag_bert.csv', header=None, index=None, sep=' ', mode='w')


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

def Doc(df,supply,m):
    model = m
    length = df.shape[0]

    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
    df['patched'] = df['patched'].map(lambda x: word_tokenize(x))
    df['simi'] = None

    # train doc
    if model == None:
        data = list(df['buggy']) + list(df['patched'])
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
        model = Doc2Vec(documents, vector_size=64, window=5, min_count=1, workers=4)
        model.save('../data/model/doc_frag.model')


    for index, row in df.iterrows():
        print('{}/{}'.format(index,length))
        if row['buggy'] == [] or row['patched'] == []:
            print('buggy or patched is []')
            continue
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

    # np.savetxt(r'../data/train_result_frag_doc.txt', df[['bugid','simi']].values, fmt='%s', header=re)
    if supply == True:
        df[['bugid', 'simi']].to_csv('../data/experiment1/train_result_frag_doc_incorrect.csv', header=None, index=None, sep=' ',
                                     mode='w')
    else:
        df[['bugid','simi']].to_csv('../data/experiment1/train_result_frag_doc.csv', header=None, index=None, sep=' ', mode='w')
    # with open('../data/train_result_doc.txt','w+') as f:
        # f.write(re)

def code2vec(df, code2vec_path):
    length = df.shape[0]
    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: my_split(x))
    df['patched'] = df['patched'].map(lambda x: my_split(x))
    # result = cosine_similarity(bc.encode(list(np.array(df_quick['buggy']))),bc.encode(list(np.array(df_quick['patched']))))
    df['simi'] = None

    print('Loading code2vec pretrained model')
    code2vec_model = KeyedVectors.load_word2vec_format(code2vec_path, binary=False)
    
    for index,row in df.iterrows():
        print('{}/{}'.format(index,length))
        if row['buggy'] == [] or row['patched'] == []:
            print('buggy or patched is []')
            continue
        try:
            buggy_tokens = []
            patch_tokens = []

            for i, b in enumerate(row['buggy']):
                if b in code2vec_model.vocab:
                    buggy_tokens.append(b)

            for i, b in enumerate(row['patched']):
                if b in code2vec_model.vocab:
                    patch_tokens.append(b)

            bug_vec = code2vec_model[buggy_tokens]
            patched_vec = code2vec_model[patch_tokens]
        except Exception as e:
            print(e)
            continue
        result = cosine_similarity(bug_vec,patched_vec)
        df.loc[index,'simi'] = float(result[0][0])
    df = df.sort_values(by='simi')
    print(df.head())
    print('the minimum similarity is {}'.format(df['simi'].head(1).values[0]))
    print('the average similarity is {}'.format(np.mean(np.array(df[['simi']]))))
    print('the median similarity is {}'.format(np.median(np.array(df[['simi']]))))

    re = ''
    # re += 'the minimum similarity is {}'.format(df['simi'].head(1).values[0]) + '\n'
    # re += 'the average similarity is {}'.format(np.mean(np.array(df[['simi']]))) + '\n'
    # re += 'the median similarity is {}'.format(np.median(np.array(df[['simi']]))) + '\n'

    # np.savetxt(r'../data/train_result_bert.txt', df[['bugid', 'simi']].values, fmt='%s', header=re)
    df[['bugid','simi']].to_csv('../data/experiment1/train_result_frag_code2vec.csv', header=None, index=None, sep=' ', mode='a+')

if __name__ == '__main__':
    # df = load_data(data_path,'patch_quicksort')
    # df = load_data(data_path_whole)

    # model = 'bert'
    models = ['doc','bert']
    # model = 'code2vec'

    for model in models:
        print('model:{}'.format(model))

        df = load_data(path_patch_train)
        core(df, model=model)

        # incorrect
        df = load_data(path_patch_train_incorrect)
        core(df, model=model,supply=True)