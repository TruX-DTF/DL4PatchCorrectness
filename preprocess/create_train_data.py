import os
import re
import pickle
import json

# path_patch = '/Users/haoye.tian/Documents/University/data/quixbugs'
# path_patched = '/Users/haoye.tian/Documents/University/data/quixbugs_wholefile/quicksort'
path_patch_test = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_test'
path_patch_train = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_train'

path_incorrect = '/Users/haoye.tian/Documents/University/data/PFL'
path_kui_data = '/Users/haoye.tian/Documents/University/data/APR-Efficiency/Patches/NFL'
path_defects4f_c = '/Users/haoye.tian/Documents/University/data/defects4j-experiment3/framework/projects'
path_supply_data = '/Users/haoye.tian/Documents/University/data/DefectRepairing/tool/patches'
data_139 = ['Patch1','Patch2','Patch4','Patch5','Patch6','Patch7','Patch8','Patch9','Patch10','Patch11','Patch12','Patch13','Patch14','Patch15','Patch16','Patch17','Patch18','Patch19','Patch20','Patch21','Patch22','Patch23','Patch24','Patch25','Patch26','Patch27','Patch28','Patch29','Patch30','Patch31','Patch32','Patch33','Patch34','Patch36','Patch37','Patch38','Patch44','Patch45','Patch46','Patch47','Patch48','Patch49','Patch51','Patch53','Patch54','Patch55','Patch58','Patch59','Patch62','Patch63','Patch64','Patch65','Patch66','Patch67','Patch68','Patch69','Patch72','Patch73','Patch74','Patch75','Patch76','Patch77','Patch78','Patch79','Patch80','Patch81','Patch82','Patch83','Patch84','Patch88','Patch89','Patch90','Patch91','Patch92','Patch93','Patch150','Patch151','Patch152','Patch153','Patch154','Patch155','Patch157','Patch158','Patch159','Patch160','Patch161','Patch162','Patch163','Patch165','Patch166','Patch167','Patch168','Patch169','Patch170','Patch171','Patch172','Patch173','Patch174','Patch175','Patch176','Patch177','Patch180','Patch181','Patch182','Patch183','Patch184','Patch185','Patch186','Patch187','Patch188','Patch189','Patch191','Patch192','Patch193','Patch194','Patch195','Patch196','Patch197','Patch198','Patch199','Patch201','Patch202','Patch203','Patch204','Patch205','Patch206','Patch207','Patch208','Patch209','Patch210','PatchHDRepair1','PatchHDRepair3','PatchHDRepair4','PatchHDRepair5','PatchHDRepair6','PatchHDRepair7','PatchHDRepair8','PatchHDRepair9','PatchHDRepair10']

bug_folder = ['Chart', 'Closure', 'Lang', 'Math', 'Time']

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



def get_diff_files(patch,type):
    with open(patch, 'r') as file:
        lines = ''
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
                        line = line.split(' ')[1]
                        lines += line.strip() + ' '
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'):
                            continue
                        lines += line[1:].strip() + ' '
                    elif line.startswith('+'):
                        # do nothing
                        pass
                    else:
                        lines += line.strip() + ' '
                elif type == 'patched':
                    if line.startswith('+++'):
                        line = line.split(' ')[1]
                        lines += line.strip() + ' '
                    elif line.startswith('+'):
                        if line[1:].strip().startswith('//'):
                            continue
                        lines += line[1:].strip() + ' '
                    elif line.startswith('-'):
                        # do nothing
                        pass
                    else:
                        lines += line.strip() + ' '
        # except Exception:
        #     print(Exception)
        #     return 'Error'
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
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index') or line.startswith('Binary'):
                    continue
                elif '/*' in line:
                    flag = False
                    continue
                elif type == 'buggy':
                    if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                        # continue
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
                    if line.startswith('PATCH_DIFF_ORIG=---'):
                        continue
                    elif line.startswith('+++'):
                        # continue
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

def get_whole(path):
    with open(path, 'r') as f:
        lines = ''
        for line in f:
            line = line.strip()
            lines += line + ' '
    return lines

def create_data(path_patch):
    path_patch_list = os.listdir(path_patch)
    path_patch_list = sorted(path_patch_list)
    with open('../data/pre_data_new.txt','w+') as f:
        data = ''
        for file_name in path_patch_list:
            bug_id = file_name
            try:
                buggy = get_diff_files(os.path.join(path_patch, file_name),type='buggy')
                patched = get_diff_files(os.path.join(path_patch, file_name),type='patched')
            except Exception:
                print(Exception)
                continue
            label_temp = '1'
            sample = label_temp + '<ml>'+ bug_id + '<ml>' + buggy + '<ml>' + patched
            data += sample + '\n'
        f.write(data)

def create_data_wholefile(path_patched):
    with open('../data/pre_data_whole.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patched):
            for file in files:
                if file == 'QUICKSORT.java':
                    buggy = get_whole('/Users/haoye.tian/Documents/University/data/quixbugs_wholefile/quicksort/QUICKSORT_BUG.java')
                    patched = get_whole(os.path.join(root, file))
                    bug_id = root.split('quicksort/')[1].split('/java_programs')[0]
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
        f.write(data)

#for future
def create_data_kui2(path_patch_kui):
    with open('../data/pre_data_kui.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patch_kui):
            for file in files:
                if file.endswith('txt') or file.endswith('patch'):
                    if file.endswith('src.patch'):
                        bug_id = '_'.join([root.split('/')[-2],root.split('/')[-1], file])
                    else:
                        bug_id = '_'.join([root.split('/')[-1],file])
                    buggy = get_diff_files(os.path.join(root,file),type='buggy')
                    patched = get_diff_files(os.path.join(root, file), type='patched')
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
                    # with open(root+'/pre_data_kui.txt', 'a+') as f:
                    #     f.write(sample)
        f.write(data)

def create_train_data5(path_patch_kui):
    with open('../data/train_data5.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            for file in files:
                # if root.startswith('../data/kui_Patches/Patches_train/Bears'):
                #     pass
                if file.endswith('txt') or file.endswith('patch'):
                    if file.endswith('.patch'):
                        bug_id = '_'.join([root.split('/')[-2],root.split('/')[-1], file])
                    else:
                        bug_id = '_'.join([root.split('/')[-1],file])
                    try:
                        # if bug_id.endswith('Bears-114.txt'):
                            # pass
                        buggy = get_diff_files(os.path.join(root,file),type='buggy')
                        patched = get_diff_files(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
                    # with open(root+'/pre_data_kui.txt', 'a+') as f:
                    #     f.write(sample)
        f.write(data)

def create_train_data5_frag(path_patch_train):
    with open('../data/experiment1/train_data5_frag_error.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patch_train):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            for file in files:
                # if root.startswith('../data/kui_Patches/Patches_train/Bears'):
                #     pass
                if file.endswith('txt') or file.endswith('patch'):
                    if file.endswith('.patch'):
                        bug_id = '_'.join([root.split('/')[-2],root.split('/')[-1], file])
                    else:
                        bug_id = '_'.join([root.split('/')[-1],file])
                    try:
                        # if bug_id.endswith('Bears-114.txt'):
                            # pass
                        buggy = get_diff_files_frag(os.path.join(root,file),type='buggy')
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    if buggy == '' or patched == '':
                        print('null patch')
                        continue
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
                    # with open(root+'/pre_data_kui.txt', 'a+') as f:
                    #     f.write(sample)
        f.write(data)

def create_train_data5_for_cc2v(path_patch_kui):
    with open('../data/experiment1/train_data5_for_cc2v.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            for file in files:
                if file.endswith('txt') or file.endswith('patch'):
                    if file.endswith('.patch'):
                        bug_id = '_'.join([root.split('/')[-2], root.split('/')[-1], file])
                    else:
                        bug_id = '_'.join([root.split('/')[-1], file])
                    try:
                        patch_cc2v = get_patch_cc2v(os.path.join(root, file))
                    except Exception as e:
                        print(e)
                        continue
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + patch_cc2v
                    data += sample + '\n'
        f.write(data)


def create_train_data_for_incorrect(path_incorrect):
    with open('../data/experiment1/train_data5_frag_incorrect.txt','w+') as f:
        data = ''

        # patch from kui
        for root, dirs, files in os.walk(path_incorrect):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            # get label
            label_temp = root.split('/')[-1]
            label = '0' if ('P' in label_temp) else '1'
            if label == '1':
                continue
            bugy_all = ''
            patched_all = ''
            for file in files:
                if file.endswith('txt'):
                    bug_id = '_'.join([root.split('/')[-1], file])
                    try:
                        buggy = get_diff_files_frag(os.path.join(root, file), type='buggy')
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    bugy_all += buggy
                    patched_all += patched
            if bugy_all == '' or patched_all == '':
                continue
            sample = label + '<ml>' + bug_id + '<ml>' + bugy_all + '<ml>' + patched_all
            data += sample + '\n'

        f.write(data)



if __name__ == '__main__':

    # create_train_data5(path_patch_train)

    # correct patches
    # create_train_data5_frag(path_patch_train)
    create_train_data5_for_cc2v(path_patch_train)

    # incorrect patches
    # create_train_data_for_incorrect(path_incorrect)
