import os
import re
import pickle
import json

path_patch_test = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_test'
benchmarks = ["Bears", "Bugs.jar", "Defects4J", "IntroClassJava", "QuixBugs"]

def get_diff_files(patch,type):
    lines = ''
    flag = True
    # try:
    for line in patch:
        line = line.strip()
        if '*/' in line:
            flag = True
            continue
        if flag == False:
            continue
        if line != '':
            if line.startswith('@@') or line.startswith('diff'):
                continue
            if line.startswith('---') or line.startswith('+++'):
                # line = line.split(' ')[1]
                # lines += line.strip() + '<nl>'
                continue
            elif '/*' in line:
                flag = False
                continue
            elif type == 'buggy':
                if line.startswith('-'):
                    if line[1:].strip().startswith('//') or line[1:].strip() == '':
                        continue
                    lines += line[1:].strip().strip('\\t') + ' '
                elif line.startswith('+'):
                    # do nothing
                    pass
                else:
                    lines += line.strip().strip('\\t') + ' '
            elif type == 'patched':
                if line.startswith('+'):
                    if line[1:].strip().startswith('//') or line[1:].strip() == '':
                        continue
                    lines += line[1:].strip().strip('\\t') + ' '
                elif line.startswith('-'):
                    # do nothing
                    pass
                else:
                    lines += line.strip().strip('\\t') + ' '
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

def get_diff_files_frag_json(patch,type):
    # with open(patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        flag = True
        # try:
        for line in patch:
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

def get_patch_cc2v(patch):
    p = r"([^\w_])"
    lines = ''
    for line in patch:
        line = line.strip()
        if line != '':
            if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                continue
            if line.startswith('---'):
                newline = re.split(pattern=p, string=line.split(' ')[1].strip())
                newline = 'mmm ' + ' '.join(newline) + ' <nl> '
                lines += newline
            elif line.startswith('+++'):
                newline = re.split(pattern=p, string=line.split(' ')[1].strip())
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
# create test data from RepairThemAll_experiment
def create_test_data(path_patch_test):
    # with open('../data/test_data.txt','w+') as f:
    #     data = ''
        for benchmark in benchmarks:
            benchmark_path = os.path.join(path_patch_test, benchmark)
            for project in os.listdir(benchmark_path):
                if project.startswith('.'):
                    continue
                project_path = os.path.join(benchmark_path, project)
                folders = os.listdir(project_path)
                if benchmark == "QuixBugs":
                    folders = [""]
                for id in folders:
                    if id.startswith('.'):
                        continue
                    bug_path = os.path.join(project_path, id)
                    data = ''
                    for repair_tool in os.listdir(bug_path):
                        tool_path = os.path.join(bug_path, repair_tool)
                        if not os.path.isdir(tool_path):
                            continue
                        for seed in os.listdir(tool_path):
                            seed_path = os.path.join(tool_path, seed)
                            results_path = os.path.join(seed_path, "result.json")
                            if os.path.exists(results_path):
                                bug_id = benchmark + '-' + project + '-' + id
                                patch_id = repair_tool + '-' + seed
                                with open(results_path,'r') as f1:
                                    patch_dict = json.load(f1)
                                patches = patch_dict['patches']
                                if patches != []:
                                    for p in patches:
                                        if 'patch' in p:
                                            patch = p['patch'].split('\n')
                                        elif 'PATCH_DIFF_ORIG' in p:
                                            patch = p['PATCH_DIFF_ORIG'].split('\\n')
                                        buggy = get_diff_files(patch, type='buggy')
                                        patched = get_diff_files(patch, type='patched')
                                        sample = '<ml>'.join(['1',bug_id,patch_id,buggy,patched,'<dl>'.join(patch)])
                                        data += sample + '\n'
                    if data != '':
                        with open(bug_path + '/test_data_bug_patches.txt', 'w+') as f2:
                            f2.write(data)

def create_test_data_frag(path_patch_test):
    # with open('../data/test_data.txt','w+') as f:
    #     data = ''
        for benchmark in benchmarks:
            benchmark_path = os.path.join(path_patch_test, benchmark)
            for project in os.listdir(benchmark_path):
                if project.startswith('.'):
                    continue
                project_path = os.path.join(benchmark_path, project)
                folders = os.listdir(project_path)
                if benchmark == "QuixBugs":
                    folders = [""]
                for id in folders:
                    if id.startswith('.'):
                        continue
                    bug_path = os.path.join(project_path, id)
                    data = ''
                    for repair_tool in os.listdir(bug_path):
                        tool_path = os.path.join(bug_path, repair_tool)
                        if not os.path.isdir(tool_path):
                            continue
                        for seed in os.listdir(tool_path):
                            seed_path = os.path.join(tool_path, seed)
                            results_path = os.path.join(seed_path, "result.json")
                            if os.path.exists(results_path):
                                bug_id = benchmark + '-' + project + '-' + id
                                patch_id = repair_tool + '-' + seed
                                with open(results_path,'r') as f1:
                                    patch_dict = json.load(f1)
                                patches = patch_dict['patches']
                                if patches != []:
                                    for p in patches:
                                        if 'patch' in p:
                                            patch = p['patch'].split('\n')
                                        elif 'PATCH_DIFF_ORIG' in p:
                                            patch = p['PATCH_DIFF_ORIG'].split('\\n')
                                        buggy = get_diff_files_frag(patch, type='buggy')
                                        patched = get_diff_files_frag(patch, type='patched')
                                        sample = '<ml>'.join(['1',bug_id,patch_id,buggy,patched,'<dl>'.join(patch)])
                                        data += sample + '\n'
                    if data != '':
                        with open(bug_path + '/test_data_bug_patches.txt', 'w+') as f2:
                            f2.write(data)

def create_test_data_for_cc2v(path_patch_test):
    with open('../data/test_data_bug_patches_all_for_cc2v.txt','w+') as f0:
        all_data = ''
        for benchmark in benchmarks:
            benchmark_path = os.path.join(path_patch_test, benchmark)
            for project in os.listdir(benchmark_path):
                if project.startswith('.'):
                    continue
                project_path = os.path.join(benchmark_path, project)
                folders = os.listdir(project_path)
                if benchmark == "QuixBugs":
                    folders = [""]
                for id in folders:
                    if id.startswith('.'):
                        continue
                    bug_path = os.path.join(project_path, id)
                    data = ''
                    for repair_tool in os.listdir(bug_path):
                        tool_path = os.path.join(bug_path, repair_tool)
                        if not os.path.isdir(tool_path):
                            continue
                        for seed in os.listdir(tool_path):
                            seed_path = os.path.join(tool_path, seed)
                            results_path = os.path.join(seed_path, "result.json")
                            if os.path.exists(results_path):
                                bug_id = benchmark + '-' + project + '-' + id
                                patch_id = repair_tool + '-' + seed
                                with open(results_path,'r') as f1:
                                    patch_dict = json.load(f1)
                                patches = patch_dict['patches']
                                if patches != []:
                                    for p in patches:
                                        if 'patch' in p:
                                            patch = p['patch'].split('\n')
                                        elif 'PATCH_DIFF_ORIG' in p:
                                            patch = p['PATCH_DIFF_ORIG'].split('\\n')
                                        try:
                                            patch_cc2v = get_patch_cc2v(patch)
                                        except Exception as e:
                                            print(e)
                                            continue

                                        label_temp = '1'
                                        sample = label_temp + '<ml>' + bug_id + '<ml>' + patch_id + '<ml>' + patch_cc2v + '<ml>' + '<dl>'.join(patch)
                                        data += sample + '\n'
                    if data != '':
                        all_data += data
                        with open(bug_path + '/test_data_bug_patches_for_cc2v.txt', 'w+') as f2:
                            f2.write(data)
        if all_data != '':
            f0.write(all_data)

def create_test_data_frag_all(path_patch_test):
    with open('../data/test_data_frag_all_for_doc.txt','w+') as f0:
        all_data = ''
        for benchmark in benchmarks:
            benchmark_path = os.path.join(path_patch_test, benchmark)
            for project in os.listdir(benchmark_path):
                if project.startswith('.'):
                    continue
                project_path = os.path.join(benchmark_path, project)
                folders = os.listdir(project_path)
                if benchmark == "QuixBugs":
                    folders = [""]
                for id in folders:
                    if id.startswith('.'):
                        continue
                    bug_path = os.path.join(project_path, id)
                    data = ''
                    for repair_tool in os.listdir(bug_path):
                        tool_path = os.path.join(bug_path, repair_tool)
                        if not os.path.isdir(tool_path):
                            continue
                        for seed in os.listdir(tool_path):
                            seed_path = os.path.join(tool_path, seed)
                            results_path = os.path.join(seed_path, "result.json")
                            if os.path.exists(results_path):
                                bug_id = benchmark + '-' + project + '-' + id
                                patch_id = repair_tool + '-' + seed
                                with open(results_path,'r') as f1:
                                    patch_dict = json.load(f1)
                                patches = patch_dict['patches']
                                if patches != []:
                                    for p in patches:
                                        if 'patch' in p:
                                            patch = p['patch'].split('\n')
                                        elif 'PATCH_DIFF_ORIG' in p:
                                            patch = p['PATCH_DIFF_ORIG'].split('\\n')

                                        # patch_cc2v = get_patch_cc2v(patch)
                                        try:
                                            buggy = get_diff_files_frag_json(patch, type='buggy')
                                            patched = get_diff_files_frag_json(patch, type='patched')
                                        except Exception as e:
                                            print(e)
                                            continue
                                        label_temp = '1'
                                        sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                                        data += sample + '\n'
                    if data != '':
                        all_data += data
        if all_data != '':
            f0.write(all_data)

if __name__ == '__main__':
    # create_test_data(path_patch_test)
    # train doc in all data
    # create_test_data_frag_all(path_patch_test)

    # not use
    # create_test_data_frag(path_patch_test)

    # collect patches for dict of cc2vec
    create_test_data_for_cc2v(path_patch_test)
