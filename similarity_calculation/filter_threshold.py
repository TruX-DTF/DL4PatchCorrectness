import os
import shutil

path_result_all = '/Users/haoye.tian/Documents/University/data/experiment2'


def filter(path_result_all, model, threshold,threshold_name,quix_thresh):
    repair_tool = os.listdir(path_result_all)
    for tool in repair_tool:
        path_tool = os.path.join(path_result_all, tool)
        bugs = os.listdir(path_tool)
        for bug in bugs:
            if bug.startswith('.'):
                continue
            path_bug = os.path.join(path_tool, bug)
            results = os.listdir(path_bug)
            for re in results:
                if re == model + '_all':
                    path_re = os.path.join(path_bug,re)
                    patches = os.listdir(path_re)
                    for patch in patches:
                        simi = float(patch.split('_')[1])
                        if bug.startswith('QuixBugs'):
                            if simi >= quix_thresh:
                                path_simi = os.path.join(path_re, patch)
                                path_simi_new = path_simi.replace('experiment2', threshold_name)
                                path_simi_new = path_simi_new.replace(re, model + '_' + threshold_name)

                                dir_new = '/'.join(path_simi_new.split('/')[:-1])
                                if not os.path.exists(dir_new):
                                    os.makedirs(dir_new)
                                shutil.copyfile(path_simi, path_simi_new)
                        else:
                            if simi >= threshold:
                                path_simi = os.path.join(path_re, patch)
                                path_simi_new = path_simi.replace('experiment2',threshold_name)
                                path_simi_new = path_simi_new.replace(re,model+'_'+threshold_name)

                                dir_new = '/'.join(path_simi_new.split('/')[:-1])
                                if not os.path.exists(dir_new):
                                    os.makedirs(dir_new)
                                shutil.copyfile(path_simi, path_simi_new)



if __name__ == '__main__':

    threshold_name = '1stqu'

    # model = 'bert'
    # threshold = 0.9947

    model = 'cc2vec'
    threshold = 0.9991

    # model = 'doc'
    # threshold = 0.8580

    # quick bugs 1stqu
    thresh_quixbugs_1stqu_bert = 0.9969
    thresh_quixbugs_1stqu_cc2vec = 0.9994
    thresh_quixbugs_1stqu_doc = 0.8956

    path_result_threshold = '/Users/haoye.tian/Documents/University/data/'+threshold_name
    filter(path_result_all=path_result_all,model=model,threshold=threshold,threshold_name=threshold_name,quix_thresh=thresh_quixbugs_1stqu_bert)