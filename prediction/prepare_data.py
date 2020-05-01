import os
import re
from gensim.models import word2vec,Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from bert_serving.client import BertClient
from nltk.tokenize import word_tokenize


path_kui_data = '/Users/haoye.tian/Documents/University/project/APR-Efficiency/Patches/NFL'
path_defects4f_c = '/Users/haoye.tian/Documents/University/data/defects4j-experiment3/framework/projects'
bug_folder = ['Chart', 'Closure', 'Lang', 'Math', 'Time']

def get_diff_files_frag(patch,type):
    with open(patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        flag = True
        # try:
        for line in file:
            line = line.strip()
            if '*/' in line:
                flag = True
                continue
            if flag == False:
                continue
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                elif '/*' in line:
                    flag = False
                    continue
                elif type == 'buggy':
                    if line.startswith('---'):
                        line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        lines += ' '.join(line) + ' '
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
                        line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        lines += ' '.join(line) + ' '
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

def get_sample(model, files, root,):

    bugy_all = ''
    patched_all = ''

    for file in files:
        if file.endswith('txt'):
            bug_id = '_'.join([root.split('/')[-2], root.split('/')[-1], file])
            try:
                buggy = get_diff_files_frag(os.path.join(root, file), type='buggy')
                patched = get_diff_files_frag(os.path.join(root, file), type='patched')
            except Exception as e:
                print(e)
                continue
            bugy_all += buggy
            patched_all += patched

    # tokenize word
    bugy_all = word_tokenize(bugy_all)
    patched_all = word_tokenize(patched_all)

    if model == 'bert':
        bug_vec, patched_vec = bert(bugy_all, patched_all)
    elif model == 'doc':
        bug_vec, patched_vec = doc(bugy_all, patched_all)
    else:
        print('wrong model')
        return

    # get label
    label_temp = root.split('/')[-1][-1]
    label = 1 if (label_temp == 'C') else 0

    return label, bug_vec, patched_vec
    # return label_array, buggy_array, patched_array

def get_sample_supply(model, path_patch,):

    bugy_all = get_diff_files_frag(path_patch, type='buggy')
    patched_all = get_diff_files_frag(path_patch, type='patched')

    # tokenize word
    bugy_all = word_tokenize(bugy_all)
    patched_all = word_tokenize(patched_all)

    if model == 'bert':
        bug_vec, patched_vec = bert(bugy_all, patched_all)
    elif model == 'doc':
        bug_vec, patched_vec = doc(bugy_all, patched_all)
    else:
        print('wrong model')
        return

    # get label
    label = 1

    return label, bug_vec, patched_vec
    # return label_array, buggy_array, patched_array

def create_kui_data_for(path_patch_kui, path_defects4f_c, model):
    print('model: {}'.format(model))
    with open('../data/experiment3/kui_data_for_'+model+'.pickle','wb') as f:

        # buggy_array = np.array([])
        # patched_array = np.array([])
        # label_array = np.array([])

        cnt = 0
        label_array, buggy_array, patched_array = list(), list(), list()

        # label=1 supply
        for bug in bug_folder:
            bug_path = os.path.join(path_defects4f_c,bug)
            correct_patches = os.path.join(bug_path,'patches')
            for patch in os.listdir(correct_patches):
                path_patch = os.path.join(correct_patches,patch)
                try:
                    label, bug_vec, patched_vec = get_sample_supply(model, path_patch,)
                    if cnt == 0:
                        buggy_array = bug_vec.reshape((1, -1))
                        patched_array = patched_vec.reshape((1, -1))
                        label_array = [label]
                    else:
                        buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
                        patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
                        label_array.append(label)
                    cnt += 1
                except Exception as e:
                    print(e)
                    continue

        # dataset
        for root,dirs,files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            label, bug_vec, patched_vec = get_sample(model, files, root,)
            if cnt == 0:
                buggy_array = bug_vec.reshape((1, -1))
                patched_array = patched_vec.reshape((1, -1))
                label_array = [label]
            else:
                buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
                patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
                label_array.append(label)
            cnt += 1

        label_array = np.array(label_array)
        data = label_array,buggy_array,patched_array
        pickle.dump(data, f)


def doc(bugy_all,patched_all):

    model = Doc2Vec.load('../data/doc_frag.model')

    bug_vec = model.infer_vector(bugy_all,alpha=0.025,steps=300)
    patched_vec = model.infer_vector(patched_all,alpha=0.025,steps=300)
    # similarity calculation
    # result = cosine_similarity(bug_vec.reshape((1,-1)), patched_vec.reshape((1,-1)))
    return bug_vec, patched_vec

def bert(bugy_all, patched_all):

    # max_seq_len=360
    bc = BertClient(check_length=False)

    bug_vec = bc.encode([bugy_all], is_tokenized=True)
    patched_vec = bc.encode([patched_all], is_tokenized=True)

    return bug_vec, patched_vec


if __name__ == '__main__':

    model = ['doc','bert']

    for i in range(len(model)):
        create_kui_data_for(path_kui_data, path_defects4f_c, model=model[i])