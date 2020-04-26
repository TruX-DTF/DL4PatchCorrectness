import numpy as np
import pandas as pd

path = '../data/experiment1/train_result_frag_doc.csv'

def calcu(path):
    df = pd.read_csv(path, sep=' ')
    df.columns = ['patch_id', 'simi']
    print('the minimum similarity is {}'.format(df['simi'].head(1).values[0]))
    print('the average similarity is {}'.format(np.mean(np.array(df[['simi']]))))
    print('the median similarity is {}'.format(np.median(np.array(df[['simi']]))))


if __name__ == '__main__':
    calcu(path)