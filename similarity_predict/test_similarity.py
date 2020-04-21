import os
import numpy as np
import pandas as pd
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec,Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


path_patch_test = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_test'

def test_similarity(path_patch_test, model, threshold):
    # os.remove('../data/test_result_'+ model + '.txt' )
    for root,dirs,files in os.walk(path_patch_test):
        for file in files:
            if file == 'test_data_bug_patches.txt':
                test_data = os.path.join(root,file)
                data = np.loadtxt(test_data, dtype=str, comments=None, delimiter='<ml>')
                data = data.reshape((-1,6))
                try:
                    df = pd.DataFrame(data, dtype=str, columns=['label', 'bugid', 'patchid','buggy', 'patched','patch'])
                except Exception as e:
                    print(e)
                # tokenize
                df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
                df['patched'] = df['patched'].map(lambda x: word_tokenize(x))
                df['simi'] = None

                if model == 'bert':
                    df_ranked = bert(df, threshold)
                elif model == 'doc':
                    df_ranked = doc(df,threshold)
                else:
                    print('wrong model')

                path_rank = os.path.join(root,model+'_top10')
                if not os.path.exists(path_rank):
                    os.mkdir(path_rank)

                for index, row in df_ranked.iterrows():
                    similarity = str(row['simi'])
                    patch_id = str(row['patchid'])
                    path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
                    patch = str(row['patch'])
                    patch = patch.replace('<dl>','\n')
                    with open(os.path.join(path_rank, path_save),'w+') as f:
                        f.write(patch)

                    # df_ranked[['bugid','patchid','simi']].to_csv(os.path.join(root,'ranked_list.csv'), header=None, index=None, sep=' ',
                    #                              mode='a')


def bert(df, threshold):
    block = ''
    length =df.shape[0]
    bug_id = str(df['bugid'][0])
    block += '**************\n'
    block += 'Budid: {}, patches: {} \n'.format(bug_id,length)

    # to do: max_seq_len=360
    bc = BertClient(check_length=False)
    for index, row in df.iterrows():
        try:
            bug_vec = bc.encode([row['buggy']], is_tokenized=True)
            patched_vec = bc.encode([row['patched']], is_tokenized=True)
        except Exception as e:
            print(e)
            continue
        result = cosine_similarity(bug_vec, patched_vec)
        df.loc[index, 'simi'] = result[0][0]
    df = df.sort_values(by='simi',ascending=False)
    df.index = range(len(df))
    # threshold
    # df = df[df['simi'].values >= threshold]

    # top filter
    top = 10
    df = df[:top]

    block += 'Threshold: {}, post_patches: {}\n'.format(threshold, df.shape[0])

    block += '{}\n'.format(df[['bugid', 'patchid', 'simi']])
    print(block)
    with open('../data/test_result_bert_new_top10.txt', 'a+') as f:
        f.write(block)
    return df

def doc(df, threshold):
    block = ''
    length = df.shape[0]
    bug_id = str(df['bugid'][0])
    block += '**************\n'
    block += 'Budid: {}, patches: {} \n'.format(bug_id, length)

    # data = list(df['buggy']) + list(df['patched'])
    # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
    # model = Doc2Vec(documents, vector_size=64, window=5, min_count=1, workers=4)
    model = Doc2Vec.load('../data/doc.model')

    for index, row in df.iterrows():
        bug_vec = model.infer_vector(row['buggy'],alpha=0.025,steps=300)
        patched_vec = model.infer_vector(row['patched'],alpha=0.025,steps=300)
        # similarity calculation
        result = cosine_similarity(bug_vec.reshape((1,-1)), patched_vec.reshape((1,-1)))
        df.loc[index, 'simi'] = result[0][0]
    df = df.sort_values(by='simi', ascending=False)
    df.index = range(len(df))
    # threshold
    # df = df[df['simi'].values >= threshold]

    # top filter
    top = 10
    df = df[:top]

    block += 'Threshold: {}, post_patches: {}\n'.format(threshold, df.shape[0])

    block += '{}\n'.format(df[['bugid', 'patchid', 'simi']])
    print(block)
    with open('../data/test_result_doc_new_top10.txt', 'a+') as f:
        f.write(block)
    return df

if __name__ == '__main__':

    # bert minumum, average, median
    # model = 'bert'
    # threshold = [0.786553263, 0.99778149, 0.998988]

    # doc minumum, average, median
    model = 'doc'
    threshold = [0.12357366, 0.944302260, 0.972037911]

    test_similarity(path_patch_test,model=model,threshold=threshold[1])
