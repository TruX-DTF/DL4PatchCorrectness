import os
import numpy as np
import pandas as pd
import re
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec,Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import json
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


path_patch_test = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_test'
root_new = '/Users/haoye.tian/Documents/University/data/experiment2'

benchmarks = ["Bears", "Bugs.jar", "Defects4J", "IntroClassJava","QuixBugs"]
# benchmarks = ['QuixBugs']
tools = ["Arja", "GenProg", "Kali", "RSRepair", "Cardumen", "jGenProg", "jKali", "jMutRepair", "Nopol", "DynaMoth", "NPEFix"]

def get_diff_files_frag(patch,type):
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

def test_similarity_repair_tool(path_patch_test, model, threshold):
    for benchmark in sorted(benchmarks):
        benchmark_path = os.path.join(path_patch_test, benchmark)
        for project in sorted(os.listdir(benchmark_path)):
            if project.startswith('.'):
                continue
            project_path = os.path.join(benchmark_path, project)
            folders = os.listdir(project_path)
            if benchmark == "QuixBugs":
                folders = [""]
            for id in sorted(folders):
                if id.startswith('.'):
                    continue
                bug_path = os.path.join(project_path, id)
                data = ''
                for repair_tool in sorted(os.listdir(bug_path)):
                    if repair_tool not in tools:
                        continue
                    tool_path = os.path.join(bug_path, repair_tool)
                    if not os.path.isdir(tool_path):
                        continue
                    for seed in sorted(os.listdir(tool_path)):
                        if type(seed).__name__ == 'list' and len(seed) > 1:
                            print('warning...')
                        seed_path = os.path.join(tool_path, seed)
                        results_path = os.path.join(seed_path, "result.json")
                        if os.path.exists(results_path):
                            bug_id = benchmark + '-' + project + '-' + id
                            patch_id = repair_tool + '-' + seed
                            with open(results_path, 'r') as f1:
                                patch_dict = json.load(f1)
                            patches = patch_dict['patches']
                            if patches != []:
                                # data = np.array([])
                                path_result = os.path.join(root_new,repair_tool,bug_id)
                                # Doc_top10 = os.path.join(path_result,'Doc_top10')
                                # Bert_top10 = os.path.join(path_result,'Bert_top10')
                                # Doc_threshold = os.path.join(path_result,'Doc_threshold')
                                # Bert_threshold = os.path.join(path_result,'Bert_threshold')
                                if os.path.exists(os.path.join(path_result, model + '_all')):
                                    continue
                                cnt = 0
                                for p in patches:
                                    if 'patch' in p:
                                        patch = p['patch']
                                        p_list = patch.split('\n')
                                    elif 'PATCH_DIFF_ORIG' in p:
                                        patch = p['PATCH_DIFF_ORIG']
                                        p_list = patch.split('\\n')
                                    else:
                                        print('error...')
                                    buggy = get_diff_files_frag(p_list, type='buggy')
                                    patched = get_diff_files_frag(p_list, type='patched')
                                    sample = np.array(['1', bug_id, patch_id, buggy, patched, patch]).reshape((-1, 6))
                                    if cnt == 0:
                                        data = sample
                                    else:
                                        data = np.concatenate((data, sample), axis=0)
                                    cnt += 1

                                calulate_similarity(path_result,data,model,threshold)


def calulate_similarity(path_result,data, model, threshold):
    data = data.reshape((-1, 6))
    df = pd.DataFrame(data, dtype=str, columns=['label', 'bugid', 'patchid', 'buggy', 'patched', 'patch'])
    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
    df['patched'] = df['patched'].map(lambda x: word_tokenize(x))
    df['simi'] = None

    if model == 'bert':
        df_threshold, df_top, df = bert(df, threshold)
    elif model == 'doc':
        df_threshold, df_top, df = doc(df, threshold)
    else:
        print('wrong model')
        raise ('wrong model')

    #all
    path_rank = os.path.join(path_result, model + '_all')
    if not os.path.exists(path_rank):
        os.makedirs(path_rank)
    for index, row in df.iterrows():
        similarity = str(row['simi'])
        patch_id = str(row['patchid'])
        path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
        patch = str(row['patch'])
        with open(os.path.join(path_rank, path_save), 'w+') as f:
            f.write(patch)

    # threshold version
    # path_rank = os.path.join(path_result, model+'_threshold')
    # if not os.path.exists(path_rank):
    #     os.makedirs(path_rank)
    # for index, row in df_threshold.iterrows():
    #     similarity = str(row['simi'])
    #     patch_id = str(row['patchid'])
    #     path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
    #     patch = str(row['patch'])
    #     with open(os.path.join(path_rank, path_save),'w+') as f:
    #         f.write(patch)

    # top version
    # path_rank = os.path.join(path_result, model + '_top10')
    # if not os.path.exists(path_rank):
    #     os.mkdir(path_rank)
    # for index, row in df_top.iterrows():
    #     similarity = str(row['simi'])
    #     patch_id = str(row['patchid'])
    #     path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
    #     patch = str(row['patch'])
    #     with open(os.path.join(path_rank, path_save), 'w+') as f:
    #         f.write(patch)

    # df[['bugid','patchid','simi','patch']].to_csv(os.path.join(path_result, model + '_all_patches.csv'), header=None, index=None, sep=' ', mode='a+')

def test_similarity(path_patch_test, model, threshold):
    # os.remove('../data/test_result_'+ model + '.txt' )
    flag = 0
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
                #
                # if str(df['bugid'][0]) == 'Defects4J-Chart-12':
                #     flag = 1
                #     continue
                # if flag == 0:
                #     continue

                if model == 'bert':
                    df_threshold, df_top = bert(df, threshold)
                elif model == 'doc':
                    df_threshold, df_top= doc(df, threshold)
                else:
                    print('wrong model')

                # threshold version
                path_rank = os.path.join(root,model+'_threshold_version')
                if not os.path.exists(path_rank):
                    os.mkdir(path_rank)

                for index, row in df_threshold.iterrows():
                    similarity = str(row['simi'])
                    patch_id = str(row['patchid'])
                    path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
                    patch = str(row['patch'])
                    patch = patch.replace('<dl>','\n')
                    with open(os.path.join(path_rank, path_save),'w+') as f:
                        f.write(patch)

                # top version
                path_rank = os.path.join(root, model + '_top_version')
                if not os.path.exists(path_rank):
                    os.mkdir(path_rank)

                for index, row in df_top.iterrows():
                    similarity = str(row['simi'])
                    patch_id = str(row['patchid'])
                    path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
                    patch = str(row['patch'])
                    patch = patch.replace('<dl>', '\n')
                    with open(os.path.join(path_rank, path_save), 'w+') as f:
                        f.write(patch)

            # df_ranked[['bugid','patchid','simi']].to_csv(os.path.join(root,'ranked_list.csv'), header=None, index=None, sep=' ',
            #                              mode='a')


def bert(df, threshold):
    block = ''
    length =df.shape[0]
    bug_id = str(df['bugid'][0])
    block += '**************\n'
    block += 'Bugid: {}, patches: {} \n'.format(bug_id,length)

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
    # df_threshold = df[df['simi'].values >= threshold]
    #
    # # top filter
    # top = 10
    # df_top = df[:top]
    #
    # block += 'Top: {}, post_patches: {}\n'.format(top, df_top.shape[0])
    # block += 'Threshold: {}, post_patches: {}\n'.format(threshold, df_threshold.shape[0])
    #
    # block += '{}\n'.format(df_top[['bugid', 'patchid', 'simi']])
    print(block)
    # with open('../data/test_result_bert_new_threshold_top.txt', 'a+') as f:
    #     f.write(block)

    # return df_threshold, df_top, df
    return None, None, df

def doc(df, threshold):
    block = ''
    length = df.shape[0]
    bug_id = str(df['bugid'][0])
    block += '**************\n'
    block += 'Bugid: {}, patches: {} \n'.format(bug_id, length)

    model = Doc2Vec.load('../data/model/doc_frag.model')

    for index, row in df.iterrows():
        bug_vec = model.infer_vector(row['buggy'],alpha=0.025,steps=300)
        patched_vec = model.infer_vector(row['patched'],alpha=0.025,steps=300)
        # similarity calculation
        result = cosine_similarity(bug_vec.reshape((1,-1)), patched_vec.reshape((1,-1)))
        df.loc[index, 'simi'] = result[0][0]
    df = df.sort_values(by='simi', ascending=False)
    df.index = range(len(df))
    # # threshold
    # df_threshold = df[df['simi'].values >= threshold]
    #
    # # top filter
    # top = 10
    # df_top = df[:top]
    #
    # block += 'Top: {}, post_patches: {}\n'.format(top, df_top.shape[0])
    # block += 'Threshold: {}, post_patches: {}\n'.format(threshold, df_threshold.shape[0])
    # block += '{}\n'.format(df_top[['bugid', 'patchid', 'simi']])

    print(block)
    # with open('../data/test_result_doc_new_threshold_top.txt', 'a+') as f:
    #     f.write(block)

    # return df_threshold, df_top, df
    return None, None, df

if __name__ == '__main__':

    model = 'bert'
    # model = 'doc'

    # test_similarity(path_patch_test,model=model,threshold=threshold[1])

    # calculate similarity of all patches
    test_similarity_repair_tool(path_patch_test,model=model,threshold=None)
