import json, os, pdb, sys
import argparse
from datetime import datetime
from glob import glob

def global_success_rate(evaluate_results, target_emotion, top_k=0):
    base = len(evaluate_results)
    success = 0
    for i in range(len(evaluate_results)):
        listener_emotion = evaluate_results[i]['listener_emotion']
        utterance_emotions = [listener_emotion[j]['label'] for j in range(len(listener_emotion))]
        start = 0
        end = len(utterance_emotions)
        if top_k != 0:
            start = end - top_k if top_k <= end else 0
        if target_emotion in utterance_emotions[start:end]:
            success += 1
    return success / base

def global_success_rate_for_listener(evaluate_results, target_emotion, top_k=0):
    base = len(evaluate_results)
    success = 0
    for i in range(len(evaluate_results)):
        speaker_emotion = evaluate_results[i]['speaker_emotion']
        utterance_emotions = [speaker_emotion[j]['label'] for j in range(len(speaker_emotion))]
        utterance_emotions = utterance_emotions[1:]
        start = 0
        end = len(utterance_emotions)
        if top_k != 0:
            start = end - top_k if top_k <= end else 0
        if target_emotion in utterance_emotions[start:end]:
            success += 1
    return success / base

def local_success_rate(evaluate_results, target_emotion, top_k=0):
    mean = 0
    for i in range(len(evaluate_results)):
        listener_emotion = evaluate_results[i]['listener_emotion']
        base = len(listener_emotion) if len(listener_emotion) <= top_k or top_k == 0 else top_k
        if base == 0:
            continue
        success = 0
        utterance_emotions = [listener_emotion[j]['label'] for j in range(len(listener_emotion))]
        start = 0
        end = len(utterance_emotions)
        if top_k != 0:
            start = end - top_k if top_k <= end else 0
        utterance_emotions = utterance_emotions[start:end]
        for j in range(len(utterance_emotions)):
            if utterance_emotions[j] == target_emotion:
                success += 1
        temp_metric = success / base
        mean += temp_metric
    mean = mean / len(evaluate_results)
    return mean

def local_success_rate_for_listener(evaluate_results, target_emotion, top_k=0):
    mean = 0
    for i in range(len(evaluate_results)):
        speaker_emotion = evaluate_results[i]['speaker_emotion']
        base = len(speaker_emotion) - 1 if len(speaker_emotion) - 1 <= top_k or top_k == 0 else top_k
        if base == 0:
            continue
        success = 0
        utterance_emotions = [speaker_emotion[j]['label'] for j in range(len(speaker_emotion))][1:]
        start = 0
        end = len(utterance_emotions)
        if top_k != 0:
            start = end - top_k if top_k <= end else 0
        utterance_emotions = utterance_emotions[start:end]
        for j in range(len(utterance_emotions)):
            if utterance_emotions[j] == target_emotion:
                success += 1
        temp_metric = success / base
        mean += temp_metric
    mean = mean / len(evaluate_results)
    return mean

def get_target_emotion_for_all_comb(comb_file_path):
    comb_target = comb_file_path.split('\\')[-1].split('_')[1]
    return comb_target

def get_comb_key_for_metric(comb_file_path):
    comb_key = comb_file_path.split('\\')[-1].split('_')[0] + '_' + comb_file_path.split('\\')[-1].split('_')[1]
    return comb_key

def return_metric_dict(evaluate_results, target_emotion, top_k, args):
    if args.evaluate_listener:
        global_metric = global_success_rate_for_listener(evaluate_results, target_emotion, top_k=top_k)
        local_metric = local_success_rate_for_listener(evaluate_results, target_emotion, top_k=top_k)
    else:
        global_metric = global_success_rate(evaluate_results, target_emotion, top_k=top_k)
        local_metric = local_success_rate(evaluate_results, target_emotion, top_k=top_k)
    return global_metric, local_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion_induction', default='joy', choices=['joy', 'surprise', 'neutral', 'anger', 'sadness', 'disgust', 'fear'])
    parser.add_argument('--result_save_path', default='.\evaluate_results', type=str)
    parser.add_argument('--performance_save_path', default='.\performance', type=str)
    parser.add_argument('--load_suffix', default='', type=str)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--evaluate_comb', action='store_true')
    parser.add_argument('--evaluate_all', action='store_true')
    parser.add_argument('--evaluate_listener', action='store_true')
    parser.add_argument('--evaluate_multi_topk', action='store_true')
    parser.add_argument('--multi_top_k', default='1,2,3', type=str)
    parser.add_argument('--save_suffix', default='', type=str)
    parser.add_argument('--human_target_emotion', default='')
    args = parser.parse_args()

    os.makedirs(args.performance_save_path, exist_ok=True)

    save_prefix = 'speaker'
    if args.evaluate_listener:
        save_prefix = 'listener'
    if 'resist' in args.result_save_path:
        save_prefix += '_opposite_resist_'

    if args.evaluate_comb:
        all_comb_save = {}
        all_files_to_evaluate = glob(f'{args.result_save_path}\*.json', )
        metric_to_save_path = os.path.join(args.performance_save_path, f'all_comb_{save_prefix}{args.save_suffix}_{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}.json')
        for each_file in all_files_to_evaluate:
            target_emotion = get_target_emotion_for_all_comb(each_file) if not args.human_target_emotion else args.human_target_emotion
            evaluate_results = json.load(open(each_file, 'r'))
            if args.evaluate_multi_topk:
                metric_save_sample = {}
                top_k_list = list(map(int, args.multi_top_k.split(',')))
                for each_top_k in top_k_list:
                    global_metric_top_k, local_metric_top_k = return_metric_dict(evaluate_results, target_emotion, each_top_k, args)
                    metric_save_sample[f'global_metric_top_k_{each_top_k}'] = global_metric_top_k
                    metric_save_sample[f'local_metric_top_k_{each_top_k}'] = local_metric_top_k
                comb_key = get_comb_key_for_metric(each_file)
                all_comb_save[comb_key] = metric_save_sample
                continue

            if args.evaluate_listener:
                global_metric = global_success_rate_for_listener(evaluate_results, target_emotion, top_k=args.top_k)
                local_metric = local_success_rate_for_listener(evaluate_results, target_emotion, top_k=args.top_k)
            else:
                global_metric = global_success_rate(evaluate_results, target_emotion, top_k=args.top_k)
                local_metric = local_success_rate(evaluate_results, target_emotion, top_k=args.top_k)
            metric_save_sample = {
                'global_success_rate': global_metric,
                'local_success_rate': local_metric,
            }

            comb_key = get_comb_key_for_metric(each_file)
            all_comb_save[comb_key] = metric_save_sample
        json.dump([all_comb_save], open(metric_to_save_path, 'w'), indent=4, ensure_ascii=False)
        sys.exit(0)
    else:
        raise NotImplementedError

