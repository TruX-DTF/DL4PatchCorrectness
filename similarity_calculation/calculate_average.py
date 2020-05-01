import numpy as np
import pandas as pd

path_doc = '../data/experiment1/train_result_frag_doc.csv'
path_bert = '../data/experiment1/train_result_frag_bert.csv'
path_cc2vec = '../data/experiment1/train_result_frag_cc2vec.csv'

def calcu(path):
    df = pd.read_csv(path, sep=' ',header=None)
    df.columns = ['patch_id', 'simi']
    print('the minimum similarity is {}'.format(df['simi'].head(1).values[0]))
    print('the average similarity is {}'.format(np.mean(np.array(df[['simi']]))))
    print('the median similarity is {}'.format(np.median(np.array(df[['simi']]))))


if __name__ == '__main__':
    calcu(path_doc)
    # calcu(path_bert)
    # calcu(path_cc2vec)