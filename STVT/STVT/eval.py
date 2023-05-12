import numpy as np
import h5py
from STVT.knapsack import knapsack

def eval_metrics(y_pred, y_true):

    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return [precision, recall, fscore]

def select_keyshots(predicted_list, video_number_list,image_name_list,target_list,args):
    data_path = './STVT/datasets/datasets/'+str(args.dataset)+".h5"
    data_file = h5py.File(data_path)

    predicted_single_video = []
    predicted_single_video_list = []
    target_single_video = []
    target_single_video_list = []
    video_single_list = list(set(video_number_list))
    eval_arr = []

    for i in range(len(image_name_list)):
        if image_name_list[i] == 1 and i!=0:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)
            predicted_single_video = []
            target_single_video = []

        predictedL = [predicted_list[i]]
        predicted_single_video += predictedL
        targetL = list(map(int, str(target_list[i])))
        target_single_video += targetL

        if i == len(image_name_list)-1:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)
    video_single_list_sort = sorted(video_single_list)
    True_all_video_len = 0
    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file['video_' + index]
        fea_sequencelen = (len(video['feature'][:])//args.sequence)*args.sequence
        True_all_video_len += fea_sequencelen

    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file['video_' + index]
        cps = video['change_points'][:]
        vidlen = int(cps[-1][1]) + 1
        weight = video['n_frame_per_seg'][:]
        fea_sequencelen = (len(video['feature'][:])//args.sequence)*args.sequence
        for ckeck_n in range(len(video_single_list_sort)):
            dif = True_all_video_len-len(predicted_list)
            if len(predicted_single_video_list[ckeck_n]) == fea_sequencelen or len(predicted_single_video_list[ckeck_n]) == fea_sequencelen-dif:
                pred_score = np.array(predicted_single_video_list[ckeck_n])
                up_rate = vidlen//len(pred_score)
                # print(up_rate)
                break
        #pred
        pred_score = upsample(pred_score, up_rate, vidlen)
        pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
        _, selected = knapsack(pred_value, weight, int(0.15 * vidlen))
        selected = selected[::-1]
        key_labels = np.zeros((vidlen,))
        for i in selected:
            key_labels[cps[i][0]:cps[i][1]] = 1
        pred_summary = key_labels.tolist()
        true_summary_arr_20 = video['user_summary'][:]
        eval_res = [eval_metrics(pred_summary, true_summary_1) for true_summary_1 in true_summary_arr_20]
        eval_res = np.mean(eval_res, axis=0).tolist() if args.dataset == "TVSum" else np.max(eval_res, axis=0).tolist()
        eval_arr.append(eval_res)
        
    return eval_arr

def upsample(down_arr, up_rate, vidlen):
    up_arr = np.zeros(vidlen)
    for i in range(len(down_arr)):
        for j in range(up_rate):
            up_arr[i * up_rate + j] = down_arr[i]

    return up_arr
