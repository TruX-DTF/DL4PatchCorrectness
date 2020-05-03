import os
import re
from gensim.models import word2vec,Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from bert_serving.client import BertClient
from nltk.tokenize import word_tokenize
import json

path_kui_data = '/Users/haoye.tian/Documents/University/data/APR-Efficiency/Patches/NFL'
path_defects4f_c = '/Users/haoye.tian/Documents/University/data/defects4j-experiment3/framework/projects'
path_supply_data = '/Users/haoye.tian/Documents/University/data/DefectRepairing/tool/patches'

bug_folder = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
bugs_folder = ['Chart', 'Closure', 'Lang', 'Math', 'Time','Mockito']

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
                if line.startswith('Index') or line.startswith('==='):
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

def get_sample(model, files, root, m):

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
        bug_vec, patched_vec = bert(bugy_all, patched_all, m)
    elif model == 'doc':
        bug_vec, patched_vec = doc(bugy_all, patched_all,m)
    else:
        print('wrong model')
        return

    # get label
    label_temp = root.split('/')[-1][-1]
    label = 1 if (label_temp == 'C') else 0

    return label, bug_vec, patched_vec
    # return label_array, buggy_array, patched_array

def get_sample_supply(model, path_patch,m):

    bugy_all = get_diff_files_frag(path_patch, type='buggy')
    patched_all = get_diff_files_frag(path_patch, type='patched')

    # tokenize word
    bugy_all = word_tokenize(bugy_all)
    patched_all = word_tokenize(patched_all)

    if model == 'bert':
        bug_vec, patched_vec = bert(bugy_all, patched_all,m)
    elif model == 'doc':
        bug_vec, patched_vec = doc(bugy_all, patched_all,m)
    else:
        print('wrong model')
        return

    # get label
    label = 1

    return label, bug_vec, patched_vec
    # return label_array, buggy_array, patched_array

def get_sample_supply2(model,path_patch, m):

    bugy_all = get_diff_files_frag(path_patch, type='buggy')
    patched_all = get_diff_files_frag(path_patch, type='patched')

    # tokenize word
    bugy_all = word_tokenize(bugy_all)
    patched_all = word_tokenize(patched_all)

    if model == 'bert':
        bug_vec, patched_vec = bert(bugy_all, patched_all, m)
    elif model == 'doc':
        bug_vec, patched_vec = doc(bugy_all, patched_all, m)
    else:
        print('wrong model')
        return

    return bug_vec, patched_vec

def create_kui_data_for(path_patch_kui, path_defects4f_c,path_supply_data, model):
    print('model: {}'.format(model))

    with open('../data/experiment3/kui_data_for_'+model+'.pickle','wb') as f:
        if model == 'doc':
            m = Doc2Vec.load('../data/doc_frag.model')
        elif model == 'bert':
            # max_seq_len=360
            m = BertClient(check_length=False)
        else:
            print('error')
        # buggy_array = np.array([])
        # patched_array = np.array([])
        # label_array = np.array([])

        cnt = 0
        label_array, buggy_array, patched_array = list(), list(), list()

        # supply data
        path_patch_supply = path_supply_data
        path_jsons = os.path.join(path_patch_supply,'INFO')
        json_files = os.listdir(path_jsons)
        for j in json_files:
            with open(os.path.join(path_jsons,j), 'r') as f1:
                info_dict = json.load(f1)
            if info_dict['project'] == 'Mockito':
                continue
            if info_dict['correctness'] == 'Correct':
                label = 1
            elif info_dict['correctness'] == 'Incorrect':
                label = 0
            else:
                continue
            path_patch = os.path.join(path_patch_supply, j.split('.')[0])

            bug_vec, patched_vec = get_sample_supply2(model,path_patch, m)
            if cnt == 0:
                buggy_array = bug_vec.reshape((1, -1))
                patched_array = patched_vec.reshape((1, -1))
                label_array = [label]
            else:
                buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
                patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
                label_array.append(label)
            cnt += 1

        # dataset
        for root, dirs, files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            if root.split('/')[-1].startswith('Mockito'):
                continue
            label, bug_vec, patched_vec = get_sample(model, files, root, m)
            if cnt == 0:
                buggy_array = bug_vec.reshape((1, -1))
                patched_array = patched_vec.reshape((1, -1))
                label_array = [label]
            else:
                buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
                patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
                label_array.append(label)
            cnt += 1

        # label=1 supply
        for bug in bug_folder:
            bug_path = os.path.join(path_defects4f_c,bug)
            correct_patches = os.path.join(bug_path,'patches')
            for patch in os.listdir(correct_patches):
                if not patch.endswith('src.patch'):
                    continue
                path_patch = os.path.join(correct_patches,patch)
                try:
                    label, bug_vec, patched_vec = get_sample_supply(model, path_patch,m)
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

        label_array = np.array(label_array)
        data = label_array,buggy_array,patched_array
        pickle.dump(data, f)

def create_kui_data_for_doc(path_patch_kui, path_defects4f_c):
    with open('../data/kui_data_frag_all_for_doc.txt','w+') as f:

        # buggy_array = np.array([])
        # patched_array = np.array([])
        # label_array = np.array([])

        cnt = 0
        label_array, buggy_array, patched_array = list(), list(), list()

        data = ''
        # label=1 supply
        for bug in bug_folder:
            bug_path = os.path.join(path_defects4f_c,bug)
            correct_patches = os.path.join(bug_path,'patches')
            for patch in os.listdir(correct_patches):
                if not patch.endswith('src.patch'):
                    continue
                path_patch = os.path.join(correct_patches,patch)
                try:
                    bugy_all = get_diff_files_frag(path_patch, type='buggy')
                    patched_all = get_diff_files_frag(path_patch, type='patched')
                except Exception as e:
                    print(e)
                    continue
                label_temp = '1'
                bug_id = 'kui_data'
                sample = label_temp + '<ml>' + bug_id + '<ml>' + bugy_all + '<ml>' + patched_all
                data += sample + '\n'

        # dataset
        for root,dirs,files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            bugy_all = ''
            patched_all = ''
            for file in files:
                if file.endswith('txt'):
                    try:
                        buggy = get_diff_files_frag(os.path.join(root, file), type='buggy')
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    bugy_all += buggy
                    patched_all += patched
                    label_temp = '1'
                    bug_id = 'kui_data'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + bugy_all + '<ml>' + patched_all
                    data += sample + '\n'

        if data != '':
            f.write(data)

def create_kui_data_for_supply(path_patch_kui, path_defects4f_c):
    with open('../data/train_data_supply_exp1.txt','w+') as f:
        data = ''

        # patch from kui
        for root, dirs, files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            # get label
            label_temp = root.split('/')[-1][-1]
            label = '1' if (label_temp == 'C') else '0'
            if label == '1':
                continue
            bugy_all = ''
            patched_all = ''
            for file in files:
                if file.endswith('txt'):
                    bug_id = '_'.join([root.split('/')[-1], file])
                    try:
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    # bugy_all += buggy
                    patched_all += patched
            bug = root.split('/')[-1].split('_')[0]
            project,id = bug.split('-')[0], bug.split('-')[1]
            buggy_location = os.path.join(path_defects4f_c,project,'patches',id+'.src.patch')
            try:
                buggy = get_diff_files_frag(buggy_location, type='buggy')
            except Exception as e:
                print(e)
                continue
            sample = label + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched_all
            data += sample + '\n'

        # patch from supply data
        path_patch_supply = path_supply_data
        path_jsons = os.path.join(path_patch_supply, 'INFO')
        json_files = os.listdir(path_jsons)
        for j in json_files:
            with open(os.path.join(path_jsons, j), 'r') as f1:
                info_dict = json.load(f1)
            if info_dict['correctness'] != 'Incorrect':
                continue
            path_patch = os.path.join(path_patch_supply, j.split('.')[0])
            try:
                patched = get_diff_files_frag(path_patch, type='patched')
            except Exception as e:
                print(e)
                continue
            project, id = info_dict['project'], info_dict['bug_id']
            buggy_location = os.path.join(path_defects4f_c, project, 'patches', id + '.src.patch')
            try:
                buggy = get_diff_files_frag(buggy_location, type='buggy')
            except Exception as e:
                print(e)
                continue
            sample = label + '<ml>' + project+'-'+id + '<ml>' + buggy + '<ml>' + patched
            data += sample + '\n'

        f.write(data)

def create_kui_data_for_cc2v_supply(path_patch_kui, path_defects4f_c):
    with open('../data/train_data_supply_exp1.txt','w+') as f:
        data = ''

        # patch from kui
        for root, dirs, files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            # get label
            label_temp = root.split('/')[-1][-1]
            label = '1' if (label_temp == 'C') else '0'
            if label == '1':
                continue
            bugy_all = ''
            patched_all = ''
            for file in files:
                if file.endswith('txt'):
                    bug_id = '_'.join([root.split('/')[-2], root.split('/')[-1], file])
                    try:
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    # bugy_all += buggy
                    patched_all += patched
            bug = root.split('/')[-1].split('_')[0]
            project,id = bug.split('-')[0], bug.split('-')[1]
            buggy_location = os.path.join(path_defects4f_c,project,'patches',id+'.src.patch')
            try:
                buggy = get_diff_files_frag(buggy_location, type='buggy')
            except Exception as e:
                print(e)
                continue
            sample = label + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched_all
            data += sample + '\n'

        # patch from supply data
        path_patch_supply = path_supply_data
        path_jsons = os.path.join(path_patch_supply, 'INFO')
        json_files = os.listdir(path_jsons)
        for j in json_files:
            with open(os.path.join(path_jsons, j), 'r') as f1:
                info_dict = json.load(f1)
            if info_dict['correctness'] != 'Incorrect':
                continue
            path_patch = os.path.join(path_patch_supply, j.split('.')[0])
            try:
                patched = get_diff_files_frag(path_patch, type='patched')
            except Exception as e:
                print(e)
                continue
            project, id = info_dict['project'], info_dict['bug_id']
            buggy_location = os.path.join(path_defects4f_c, project, 'patches', id + '.src.patch')
            try:
                buggy = get_diff_files_frag(buggy_location, type='buggy')
            except Exception as e:
                print(e)
                continue
            sample = label + '<ml>' + project+'-'+id + '<ml>' + buggy + '<ml>' + patched
            data += sample + '\n'

        f.write(data)

def doc(bugy_all,patched_all, model):

    bug_vec = model.infer_vector(bugy_all,alpha=0.025,steps=300)
    patched_vec = model.infer_vector(patched_all,alpha=0.025,steps=300)
    # similarity calculation
    # result = cosine_similarity(bug_vec.reshape((1,-1)), patched_vec.reshape((1,-1)))
    return bug_vec, patched_vec

def bert(bugy_all, patched_all, m):

    bug_vec = m.encode([bugy_all], is_tokenized=True)
    patched_vec = m.encode([patched_all], is_tokenized=True)

    return bug_vec, patched_vec


if __name__ == '__main__':
    # provide train data for doc
    # create_kui_data_for_doc(path_kui_data, path_defects4f_c)

    create_kui_data_for_supply(path_kui_data, path_defects4f_c)
    # create_kui_data_for_cc2v_supply(path_kui_data, path_defects4f_c)

    model = ['bert']

    # for i in range(len(model)):
    #     create_kui_data_for(path_kui_data, path_defects4f_c,path_supply_data, model=model[i])