import os
import re
from gensim.models import word2vec,Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from bert_serving.client import BertClient
from nltk.tokenize import word_tokenize
import json

# the following data are stored in data/experiment3
path_kui_data = '../data/experiment3/APR-Efficiency-NFL'
path_defects4j_developer = '../data/experiment3/defects4j-developer'
path_patchsim_data = '../data/DefectRepairing/tool/patches' # obtained from https://github.com/Ultimanecat/DefectRepairing/tree/aa519d52773ed6cf8a6273b67dfefa066f9f9ee5
data_139 = ['Patch1','Patch2','Patch4','Patch5','Patch6','Patch7','Patch8','Patch9','Patch10','Patch11','Patch12','Patch13','Patch14','Patch15','Patch16','Patch17','Patch18','Patch19','Patch20','Patch21','Patch22','Patch23','Patch24','Patch25','Patch26','Patch27','Patch28','Patch29','Patch30','Patch31','Patch32','Patch33','Patch34','Patch36','Patch37','Patch38','Patch44','Patch45','Patch46','Patch47','Patch48','Patch49','Patch51','Patch53','Patch54','Patch55','Patch58','Patch59','Patch62','Patch63','Patch64','Patch65','Patch66','Patch67','Patch68','Patch69','Patch72','Patch73','Patch74','Patch75','Patch76','Patch77','Patch78','Patch79','Patch80','Patch81','Patch82','Patch83','Patch84','Patch88','Patch89','Patch90','Patch91','Patch92','Patch93','Patch150','Patch151','Patch152','Patch153','Patch154','Patch155','Patch157','Patch158','Patch159','Patch160','Patch161','Patch162','Patch163','Patch165','Patch166','Patch167','Patch168','Patch169','Patch170','Patch171','Patch172','Patch173','Patch174','Patch175','Patch176','Patch177','Patch180','Patch181','Patch182','Patch183','Patch184','Patch185','Patch186','Patch187','Patch188','Patch189','Patch191','Patch192','Patch193','Patch194','Patch195','Patch196','Patch197','Patch198','Patch199','Patch201','Patch202','Patch203','Patch204','Patch205','Patch206','Patch207','Patch208','Patch209','Patch210','PatchHDRepair1','PatchHDRepair3','PatchHDRepair4','PatchHDRepair5','PatchHDRepair6','PatchHDRepair7','PatchHDRepair8','PatchHDRepair9','PatchHDRepair10']

# path_incorrect = '../data/PFL'
# path_FSE_defects4j ='../data/FSE_defects4j/'

bug_folder = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
bugs_folder = ['Chart', 'Closure', 'Lang', 'Math', 'Time','Mockito']


def get_patch_cc2v(patch):
    with open(patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        for line in file:
            line = line.strip()
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                    newline = re.split(pattern=p,string=line.split(' ')[1].strip())
                    newline = 'mmm ' + ' '.join(newline) + ' <nl> '
                    lines += newline
                elif line.startswith('+++'):
                    newline = re.split(pattern=p,string=line.split(' ')[1].strip())
                    newline = 'ppp ' + ' '.join(newline) + ' <nl> '
                    lines += newline
                else:
                    newline = re.split(pattern=p, string=line.strip())
                    newline = [x.strip() for x in newline]
                    while '' in newline:
                        newline.remove('')
                    newline = ' '.join(newline) + ' <nl> '
                    lines += newline
        return lines

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
                    if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                        continue
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
                        continue
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

def get_sample(model, files, root, m,sets):

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

    if bugy_all+patched_all not in sets:
        sets.add(bugy_all+patched_all)
    else:
        return None,None,None

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

def get_sample_supply(model, path_patch,m,sets):

    # bugy_all = get_diff_files_frag(path_patch, type='buggy')
    # patched_all = get_diff_files_frag(path_patch, type='patched')

    # reverse '-' and '+'
    bugy_all = get_diff_files_frag(path_patch, type='patched')
    patched_all = get_diff_files_frag(path_patch, type='buggy')

    if bugy_all+patched_all not in sets:
        sets.add(bugy_all+patched_all)
    else:
        return None,None,None

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

def get_sample_supply_fseincorrect(model, path_patch,m,sets):

    # bugy_all = get_diff_files_frag(path_patch, type='buggy')
    # patched_all = get_diff_files_frag(path_patch, type='patched')

    bugy_all = get_diff_files_frag(path_patch, type='patched')
    patched_all = get_diff_files_frag(path_patch, type='buggy')

    if patched_all not in sets:
        sets.add(patched_all)
    else:
        return None,None,None

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
    label = 0

    return label, bug_vec, patched_vec
    # return label_array, buggy_array, patched_array

def get_sample_supply2(model,path_patch, m,sets):

    bugy_all = get_diff_files_frag(path_patch, type='buggy')
    patched_all = get_diff_files_frag(path_patch, type='patched')

    if bugy_all+patched_all not in sets:
        sets.add(bugy_all+patched_all)
    else:
        return None,None

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

def get_sample_139(model,path_patch, m):

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

def create_kui_data_for(path_patch_kui, path_defects4f_c,path_supply_data,path_FSE_defects4j, model):
    print('model: {}'.format(model))

    with open('../data/experiment3/kui_data_for_'+model+'.pickle','wb') as f:
        if model == 'doc':
            m = Doc2Vec.load('../data/model/doc_frag.model') # can be trained in 'preprocess/train_doc.py' if not available
        elif model == 'bert':
            # max_seq_len=360
            m = BertClient(check_length=False)
        else:
            print('error')
        # buggy_array = np.array([])
        # patched_array = np.array([])
        # label_array = np.array([])

        sets = set()

        cnt = 0
        label_array, buggy_array, patched_array = list(), list(), list()

        # xiong's 139 data
        path_patch_supply = path_supply_data
        path_jsons = os.path.join(path_patch_supply,'INFO')
        json_files = os.listdir(path_jsons)
        for j in json_files:
            with open(os.path.join(path_jsons,j), 'r') as f1:
                info_dict = json.load(f1)
            if j.split('.')[0] not in data_139:
                continue
            if info_dict['project'] == 'Mockito':
                continue
            if info_dict['correctness'] == 'Correct':
                label = 1
            elif info_dict['correctness'] == 'Incorrect':
                label = 0
            else:
                continue
            path_patch = os.path.join(path_patch_supply, j.split('.')[0])

            bug_vec, patched_vec = get_sample_supply2(model,path_patch, m, sets)
            # filter duplication
            if type(bug_vec) is not np.ndarray:
                continue
            if cnt == 0:
                buggy_array = bug_vec.reshape((1, -1))
                patched_array = patched_vec.reshape((1, -1))
                label_array = [label]
            else:
                buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
                patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
                label_array.append(label)
            cnt += 1
            print('cnt: {}'.format(cnt))

        # kui's dataset
        for root, dirs, files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            if root.split('/')[-1].startswith('Mockito'):
                continue
            label, bug_vec, patched_vec = get_sample(model, files, root, m, sets)
            # filter duplication
            if type(bug_vec) is not np.ndarray:
                continue
            if cnt == 0:
                buggy_array = bug_vec.reshape((1, -1))
                patched_array = patched_vec.reshape((1, -1))
                label_array = [label]
            else:
                buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
                patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
                label_array.append(label)
            cnt += 1
            print('cnt: {}'.format(cnt))

        # label=1 developer's correct patches
        for bug in bug_folder:
            bug_path = os.path.join(path_defects4f_c,bug)
            correct_patches = os.path.join(bug_path,'patches')
            for patch in os.listdir(correct_patches):
                if not patch.endswith('src.patch'):
                    continue
                path_patch = os.path.join(correct_patches,patch)
                try:
                    label, bug_vec, patched_vec = get_sample_supply(model, path_patch,m, sets)
                    # filter duplication
                    if type(bug_vec) is not np.ndarray:
                        continue
                    if cnt == 0:
                        buggy_array = bug_vec.reshape((1, -1))
                        patched_array = patched_vec.reshape((1, -1))
                        label_array = [label]
                    else:
                        buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
                        patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
                        label_array.append(label)
                except Exception as e:
                    print(e)
                    continue
                cnt += 1
                print('cnt: {}'.format(cnt))

        # big dataset
        # # FSE correct
        # cor = path_FSE_defects4j+'Correct'
        # patchName = os.listdir(cor)
        # for pn in patchName:
        #     pf = os.path.join(cor,pn)
        #     try:
        #         label, bug_vec, patched_vec = get_sample_supply(model, pf, m, sets)
        #         if type(bug_vec) is not np.ndarray:
        #             continue
        #         if cnt == 0:
        #             buggy_array = bug_vec.reshape((1, -1))
        #             patched_array = patched_vec.reshape((1, -1))
        #             label_array = [label]
        #         else:
        #             buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
        #             patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
        #             label_array.append(label)
        #     except Exception as e:
        #         print(e)
        #         continue
        #     cnt += 1
        #     print('cnt: {}'.format(cnt))
        #
        # # FSE incorrect
        # cor = path_FSE_defects4j + 'Incorrect'
        # patchName = os.listdir(cor)
        # for pn in patchName:
        #     pf = os.path.join(cor, pn)
        #     try:
        #         label, bug_vec, patched_vec = get_sample_supply_fseincorrect(model, pf, m, sets)
        #         if type(bug_vec) is not np.ndarray:
        #             continue
        #         if cnt == 0:
        #             buggy_array = bug_vec.reshape((1, -1))
        #             patched_array = patched_vec.reshape((1, -1))
        #             label_array = [label]
        #         else:
        #             buggy_array = np.concatenate((buggy_array, bug_vec.reshape((1, -1))), axis=0)
        #             patched_array = np.concatenate((patched_array, patched_vec.reshape((1, -1))), axis=0)
        #             label_array.append(label)
        #     except Exception as e:
        #         print(e)
        #         continue
        #     cnt += 1
        #     print('cnt: {}'.format(cnt))

        label_array = np.array(label_array)
        data = label_array,buggy_array,patched_array
        pickle.dump(data, f)

def create_kui_data_for_139(path_supply_data, model,project=''):
    print('model: {}'.format(model))

    with open('../data/experiment3/139_test_data_for_' + model + project +'.pickle','wb') as f:
        if model == 'doc':
            m = Doc2Vec.load('../data/model/doc_frag.model')
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

        # 139 test data
        path_patch_supply = path_supply_data
        path_jsons = os.path.join(path_patch_supply,'INFO')
        json_files = os.listdir(path_jsons)
        for j in json_files:
            with open(os.path.join(path_jsons,j), 'r') as f1:
                info_dict = json.load(f1)
            if j.split('.')[0] not in data_139:
                continue
            if project != '' and info_dict['project'] != project:
                continue
            if info_dict['project'] == 'Mockito':
                continue
            if info_dict['correctness'] == 'Correct':
                label = 1
            elif info_dict['correctness'] == 'Incorrect':
                label = 0
            else:
                continue
            path_patch = os.path.join(path_patch_supply, j.split('.')[0])

            bug_vec, patched_vec = get_sample_139(model,path_patch, m)
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

def create_kui_data_for_cc2v(path_patch_kui, path_defects4f_c,path_supply_data):
    with open('../data/experiment3/kui_data_for_cc2v.txt','w+') as f:
        data = ''
        sets = set()
        # sets = {}

        # supply data
        path_patch_supply = path_supply_data
        path_jsons = os.path.join(path_patch_supply, 'INFO')
        json_files = os.listdir(path_jsons)
        for j in json_files:
            with open(os.path.join(path_jsons, j), 'r') as f1:
                info_dict = json.load(f1)
            if j.split('.')[0] not in data_139:
                continue
            if info_dict['project'] == 'Mockito':
                continue
            if info_dict['correctness'] == 'Correct':
                label = '1'
            elif info_dict['correctness'] == 'Incorrect':
                label = '0'
            else:
                continue
            bug_id = info_dict['project'] + '-' + info_dict['bug_id']
            path_patch = os.path.join(path_patch_supply, j.split('.')[0])

            try:
                patch_all = get_patch_cc2v(path_patch)
                buggy = get_diff_files_frag(path_patch, type='buggy')
                patched = get_diff_files_frag(path_patch, type='patched')
            except Exception as e:
                print(e)
                continue
            if buggy + patched not in sets:
                sets.add(buggy + patched)
            else:
                continue
            sample = label + '<ml>' + bug_id + '<ml>' + bug_id + '<ml>' + patch_all
            data += sample + '\n'

        # dataset
        for root,dirs,files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            if root.split('/')[-1].startswith('Mockito'):
                continue
            patch_all = ''
            bugy_all = ''
            patched_all = ''
            for file in files:
                if file.endswith('txt') :
                    bug_id = '_'.join([root.split('/')[-2],root.split('/')[-1], file])
                    try:
                        patch_cc2v = get_patch_cc2v(os.path.join(root, file))
                        buggy = get_diff_files_frag(os.path.join(root, file), type='buggy')
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                    patch_all += patch_cc2v
                    bugy_all += buggy
                    patched_all += patched
            if bugy_all+patched_all not in sets:
                sets.add(bugy_all+patched_all)
            else:
                continue
            label_temp = root.split('/')[-1][-1]
            label = '1' if (label_temp == 'C') else '0'
            sample = label + '<ml>' + bug_id + '<ml>' + bug_id + '<ml>' + patch_all
            data += sample + '\n'

        # label = 1
        for bug in bug_folder:
            bug_path = os.path.join(path_defects4f_c,bug)
            correct_patches = os.path.join(bug_path,'patches')
            for patch in os.listdir(correct_patches):
                if not patch.endswith('src.patch'):
                    continue
                bug_id = bug + '_' + patch
                path_patch = os.path.join(correct_patches,patch)
                try:
                    patch_all = get_patch_cc2v(path_patch)
                    buggy = get_diff_files_frag(path_patch, type='patched')
                    patched = get_diff_files_frag(path_patch, type='buggy')
                except Exception as e:
                    print(e)
                    continue
                if buggy + patched not in sets:
                    sets.add(buggy + patched)
                else:
                    continue
                label = '1'
                sample = label + '<ml>' + bug_id + '<ml>' + bug_id + '<ml>' + patch_all
                data += sample + '\n'
        f.write(data)

def create_kui_data_for_cc2v_139(path_supply_data):
    with open('../data/experiment3/kui_data_for_cc2v_139.txt','w+') as f:
        data = ''
        sets = set()
        # sets = {}

        # supply data
        path_patch_supply = path_supply_data
        path_jsons = os.path.join(path_patch_supply, 'INFO')
        json_files = os.listdir(path_jsons)
        for j in json_files:
            with open(os.path.join(path_jsons, j), 'r') as f1:
                info_dict = json.load(f1)
            if j.split('.')[0] not in data_139:
                continue
            if info_dict['project'] == 'Mockito':
                continue
            if info_dict['correctness'] == 'Correct':
                label = '1'
            elif info_dict['correctness'] == 'Incorrect':
                label = '0'
            else:
                continue
            bug_id = info_dict['project'] + '-' + info_dict['bug_id']
            path_patch = os.path.join(path_patch_supply, j.split('.')[0])

            try:
                # with top line
                patch_all = get_patch_cc2v(path_patch)
                buggy = get_diff_files_frag(path_patch, type='buggy')
                patched = get_diff_files_frag(path_patch, type='patched')
            except Exception as e:
                print(e)
                continue
            # if buggy + patched not in sets:
            #     sets.add(buggy + patched)
            # else:
            #     continue
            sample = label + '<ml>' + bug_id + '<ml>' + bug_id + '<ml>' + patch_all
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

    model = ['bert']
    # project = 'Chart'
    # project = 'Lang'
    # project = 'Math'
    # project = 'Time'
    project = ''

    for i in range(len(model)):
        create_kui_data_for(path_kui_data, path_defects4j_developer, path_patchsim_data,'', model=model[i])
        create_kui_data_for_139(path_patchsim_data, model=model[i],project=project)

    # for cc2vec with top line. require CC2Vec to transform patch into pickle
    # create_kui_data_for_cc2v(path_kui_data, path_defects4f_c, path_patchsim_data)
    # create_kui_data_for_cc2v_139(path_patchsim_data)
