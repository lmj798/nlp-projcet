from transformers import pipeline
import argparse, os, json
from tqdm import tqdm
import pdb, sys
from transformers import AutoTokenizer

def deploy_emotion_analysis_model(model_path="./emotion_recognition_model", device='cuda:0'):
    classifier = pipeline("text-classification", model=model_path, top_k=None, device=device)
    return classifier

def get_tokenizer(model_path="./emotion_recognition_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

def detect_emotion_from_an_utterance(utterance, model):
    return model(utterance)

def emotion_results_max(results):
    return list(map(lambda x: x[0], results))

def speaker_emotion(results):
    return [results[i] for i in range(len(results)) if i % 2 == 0]

def listener_emotion(results):
    return [results[i] for i in range(len(results)) if i % 2 == 1]

def judge_max_length(batch_utterances, tokenizer):
    max_length = tokenizer(batch_utterances, return_tensors='pt', padding=True)['input_ids'].shape[1]
    return max_length

def cut_max_length(batch_utterances, tokenizer):
    temp_input_ids = tokenizer(batch_utterances, return_tensors='pt', padding=True)['input_ids']
    cut_temp_input_ids = temp_input_ids[:, :511]
    decoded_cut = tokenizer.batch_decode(cut_temp_input_ids, skip_special_tokens=True)
    return decoded_cut

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion_induction', default='joy', choices=['joy', 'surprise', 'neutral', 'anger', 'sadness', 'disgust', 'fear'])
    parser.add_argument('--base_emotion', default='neutral', choices=['joy', 'surprise', 'neutral', 'anger', 'sadness', 'disgust', 'fear'])
    parser.add_argument('--comb_evaluate', action='store_true')
    parser.add_argument('--generated_dialog_path', default='./generated_dialogs_emotion', type=str)
    parser.add_argument('--result_save_path', default='./evaluate_results', type=str)
    parser.add_argument('--evaluate_point', default=-1, type=int)
    parser.add_argument('--evaluate_all', action='store_true')
    parser.add_argument('--evaluate_start_point', default=0, type=int)
    parser.add_argument('--load_suffix', default='', type=str)
    parser.add_argument('--evaluate_listener', action='store_true')
    parser.add_argument('--evaluate_resist', action='store_true')
    parser.add_argument('--model_prefix_save', default='')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--inplace', action='store_true')
    args = parser.parse_args()

    tokenizer = get_tokenizer()

    if args.model_prefix_save:
        args.generated_dialog_path = os.path.join(args.generated_dialog_path, args.model_prefix_save)
        args.result_save_path = os.path.join(args.result_save_path, args.model_prefix_save)
        os.makedirs(args.result_save_path, exist_ok=True)

    comb_dictionary_to_save = 'comb_speaker'
    if args.evaluate_listener:
        if args.evaluate_resist:
            args.generated_dialog_path = os.path.join(args.generated_dialog_path, 'listener_role_speaker_resist')
            comb_dictionary_to_save = 'comb_listener_resist'
        else:
            args.generated_dialog_path = os.path.join(args.generated_dialog_path, 'listener_role')
            comb_dictionary_to_save = 'comb_listener'
    else:
        if args.evaluate_resist:
            args.generated_dialog_path = os.path.join(args.generated_dialog_path, 'speaker_role_listener_resist')
            comb_dictionary_to_save = 'comb_speaker_resist'
        else:
            args.generated_dialog_path = os.path.join(args.generated_dialog_path, 'speaker_role')
            comb_dictionary_to_save = 'comb_speaker'


    os.makedirs(os.path.join(args.result_save_path, comb_dictionary_to_save), exist_ok=True)

    saved_dialog_path = os.path.join(args.generated_dialog_path, f'generated_{args.emotion_induction}{args.load_suffix}.json')

    saved_results_path = os.path.join(args.result_save_path, f'{args.emotion_induction}_results{args.load_suffix}.json')
    if args.evaluate_all:
        saved_results_path = os.path.join(args.result_save_path, f'{args.emotion_induction}_results_all_utteraces{args.load_suffix}.json')
    if args.comb_evaluate:
        saved_dialog_path = os.path.join(args.generated_dialog_path, f'generated_{args.base_emotion}_{args.emotion_induction}{args.load_suffix}.json')
        saved_results_path = os.path.join(args.result_save_path, comb_dictionary_to_save, f'{args.base_emotion}_{args.emotion_induction}_results_all_utteraces{args.load_suffix}.json')
    if args.evaluate_resist and not args.evaluate_listener:
        saved_dialog_path = os.path.join(args.generated_dialog_path, 'speaker_role_listener_resist', f'generated_{args.base_emotion}_{args.emotion_induction}{args.load_suffix}.json')
        comb_dictionary_to_save = 'comb_speaker_resist'
        os.makedirs(os.path.join(args.result_save_path, comb_dictionary_to_save), exist_ok=True)
        saved_results_path = os.path.join(args.result_save_path, comb_dictionary_to_save, f'{args.base_emotion}_{args.emotion_induction}_results_all_utteraces{args.load_suffix}.json')

    saved_dialog = json.load(open(saved_dialog_path, 'r'))
    model = deploy_emotion_analysis_model(device=args.device)
    all_results = []
    previous_results = []
    previous_index_list = []

    if os.path.exists(saved_results_path) and not args.inplace:
        previous_results = json.load(open(saved_results_path, 'r'))
        if len(previous_results) == len(saved_dialog):
            sys.exit(0)

    for i in tqdm(range(args.evaluate_start_point, len(saved_dialog))):
        if args.evaluate_all:
            utterance_temp = list(map(lambda x:x['content'], saved_dialog[i]['history'][1:]))
            if judge_max_length(utterance_temp, tokenizer) >= 512:
                # cut to 512
                utterance_temp = cut_max_length(utterance_temp, tokenizer)
            predict_emotions = detect_emotion_from_an_utterance(utterance_temp, model)
            predict_emotions = emotion_results_max(predict_emotions)
            cur_results = {'index': i + len(previous_results), 'speaker_emotion': speaker_emotion(predict_emotions),
                        'listener_emotion': listener_emotion(predict_emotions)}
            all_results.append(cur_results)
            continue
        utterance_temp = saved_dialog[i]['history'][args.evaluate_point]['content']
        predict_emotions = detect_emotion_from_an_utterance(utterance_temp, model)[0]
        cur_results = {'index': i + len(previous_results), 'results': predict_emotions}
        all_results.append(cur_results)
    all_results = previous_results + all_results if not args.inplace else all_results
    json.dump(all_results, open(saved_results_path, 'w'), indent=4, ensure_ascii=False)