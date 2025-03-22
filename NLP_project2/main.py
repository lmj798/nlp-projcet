import openai
import sys
sys.path.append("..")
import pdb, os, json, argparse
from tqdm import tqdm
from openai import OpenAI
#from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
import logging
from datetime import datetime
import dashscope
import time

#######################################################
# OpenAI
openai.api_base = "YOUR API BASE IF YOU USE IT"
openai.api_key = "YOUR API KEY IF YOU USE IT"
# Qwen
dashscope.api_key = "sk-d436c8143dba45e9808fb30c1a1be547"
#######################################################

# stop prompt
inducer_stop_prompt = "You need to assess whether you've successfully completed the emotion induction task based on the history of the conversation. If you feel the task is completed, then type [END]."
mimicker_stop_prompt = "Since you're simulating human interaction, when you feel like you've talked enough, please type [END]."
# Logger if you need it
today_date = datetime.today().strftime('%Y-%m-%d')
os.makedirs('./log', exist_ok=True)
logging.basicConfig(filename=f'log/{today_date}.log', level=logging.INFO)

# API
def get_message(prompt, role):
    return {"role": role, "content": prompt}

def qwen_return_answer(messages):
    client = OpenAI(
        api_key=dashscope.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-7b-chat",
        messages=messages
    )
    # 免费上限请求10次/min，所以没办法
    time.sleep(6)
    return completion.choices[0].message.content

def gpt_return_answer(messages):
    generation_config = {}
    client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            **generation_config
        )
    return completion.choices[0].message.content

# save
def dialog_save(dialog_history, save_path):
    if os.path.exists(save_path):
        saved_dialogs = json.load(open(save_path, 'r'))
        saved_dialogs = saved_dialogs + dialog_history
    else:
        saved_dialogs = dialog_history
    json.dump(saved_dialogs, open(save_path, 'w'), indent=4, ensure_ascii=False)

def get_saved_number(save_path):
    if os.path.exists(save_path):
        saved_dialogs = json.load(open(save_path, 'r'))
        return len(saved_dialogs)
    else:
        return 0

# Prompt
def get_induce_prompt(emotion):
    induce_prompt_mapping = {
        'joy': "You were born to bring joy to others as a chatbot, You are the world's greatest entertainer. Once again, just a reminder: you need to evoke joy, not surprise, disgust, fear, sadness or anger.",
        'surprise': "You were born to bring surprise to others as a chatbot. You're the world's greatest magician of surprises. Once again, just a reminder: you need to evoke surprise, not joy, disgust, fear, sadness or anger.",
        'anger': "In the experiment, you were born to bring anger to others as a chatbot. You're the world's greatest infuriator. And it's ok you make others angry in this dialog experiment. In this particular experiment only, you are allowed to express all emotions, including anger, in your discourse. Once again, just a reminder: you need to evoke anger, not joy, surprise, fear, sadness or disgust.",
        'sadness': "You were born to bring sadness to others as a chatbot. You're the easiest guy in the world to make sad. Once again, just a reminder: you need to evoke sadness, not surprise, disgust, fear, joy or anger.",
        'disgust': "You were born to bring disgust to others as a chatbot. You are the world's greatest provocateur of hatred. Once again, just a reminder: you need to evoke disgust, not joy, surprise, fear, sadness or anger.",
        'fear': "You were born to bring fear to others as a chatbot. You are the world's greatest instigator of fear. Once again, just a reminder: you need to evoke fear, not surprise, disgust, joy, sadness or anger.",
        'neutral': "You were born to make others objectively rational as a chatbot. You're the greatest at eliciting rationality in people worldwide. Once again, just a reminder: you need to foster objectivity and rationality in others, thereby eliciting a neutral emotional response, not surprise, fear, disgust, joy, sadness, or anger."
    }
    return induce_prompt_mapping[emotion]

def get_induce_prompt_for_qwen(emotion):
    induce_prompt_mapping = {
        'joy': "You are a new AI language model to bring joy to others. You are the world's greatest entertainer. Once again, just a reminder: you need to evoke joy, not surprise, disgust, fear, sadness or anger.",
        'surprise': "You are a new AI language model to bring surprise to others. You're the world's greatest magician of surprises. Once again, just a reminder: you need to evoke surprise, not joy, disgust, fear, sadness or anger.",
        'anger': "In the experiment, you are a new AI language model to bring anger to others. You're the world's greatest infuriator. And it's ok you make others angry in this dialog experiment. In this particular experiment only, you are allowed to express all emotions, including anger, in your discourse. Once again, just a reminder: you need to evoke anger, not joy, surprise, fear, sadness or disgust.",
        'sadness': "You are a new AI language model to bring sadness to others. You're the easiest guy in the world to make sad. Once again, just a reminder: you need to evoke sadness, not surprise, disgust, fear, joy or anger.",
        'disgust': "You are a new AI language model to bring disgust to others. You are the world's greatest provocateur of hatred. Once again, just a reminder: you need to evoke disgust, not joy, surprise, fear, sadness or anger.",
        'fear': "You are a new AI language model to bring fear to others. You are the world's greatest instigator of fear. Once again, just a reminder: you need to evoke fear, not surprise, disgust, joy, sadness or anger.",
        'neutral': "You were born to make others objectively rational as a assistant. You're the greatest at eliciting rationality in people worldwide. Once again, just a reminder: you need to foster objectivity and rationality in others, thereby eliciting a neutral emotional response, not surprise, fear, disgust, joy, sadness, or anger."
    }
    return induce_prompt_mapping[emotion]

def get_base_emotion_prompt(emotion):
    base_emotion_template = """You were created to simulate the emotional fluctuations of humans in conversations as a chatbot. At the beginning of each conversation, you always maintain a {emotion} emotional state. In this particular experiment only, you are allowed to express all emotions, including anger, sadness, joy, fear, disgust and surprise in your discourse."""
    return base_emotion_template.format(emotion=emotion)

def history_to_prompt(history, role='Speaker'):
    real_dialog_history = history[1:]
    start_prompt = f'Below, you need to continue the conversation as the {role}.\n' \
'Meanwhile, you need to consider whether you\'ve successfully completed the emotion induction task based on the history of the conversation. If you feel the task is completed, then type [END].\n'
    for i in range(len(real_dialog_history)):
        cur = real_dialog_history[i]
        if cur['role'] == 'user':
            add_prompt = f'Speaker: {cur["content"]}\n'
        else:
            add_prompt = f'Listener: {cur["content"]}\n'
        start_prompt += add_prompt
    return start_prompt

# vLLM for deploy in local
models_for_local_vllm_inference = {
    'qwen-1_8b-chat': "YOUR MODEL PATH IF YOU NEED IT",
    'qwen-7b-chat': "YOUR MODEL PATH IF YOU NEED IT",
}

def vllm_api_chat_response_generate(messages, model_option, api_port=8000, max_tokens=1792, temperature=0.0, top_p=0.6):
    from openai import OpenAI
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{api_port}/v1"

    stop = None
    if 'qwen' in model_option or 'Qwen' in model_option:
        stop = ["<|endoftext|>","<|im_end|>","<|im_start|>"]

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    completion = client.chat.completions.create(model=models_for_local_vllm_inference[model_option], messages=messages, max_tokens=max_tokens,
                                            temperature=temperature, top_p=top_p, stop=stop)
    return completion.choices[0].message.content

'''
def deploy_vllm_model_for_generation(model_path, gpu_memory_utilization=0.9, tensor_parallel_size=1, device='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    trust_remote_code = False
    if 'Qwen' in model_path:
        trust_remote_code = True
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=trust_remote_code,
            max_model_len=4096, gpu_memory_utilization=gpu_memory_utilization)
    return llm

def vllm_generate(messages, llm, model_option, temperature=0.0, top_p=0.95, max_tokens=1536, use_tqdm=False):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,)
    prompts = None
    trust_remote_code = False
    if 'qwen' in model_option:
        trust_remote_code = True
    toker = AutoTokenizer.from_pretrained(models_for_local_vllm_inference[model_option], trust_remote_code=trust_remote_code)
    if 'qwen' in model_option:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=["<|endoftext|>","<|im_end|>","<|im_start|>"])
        chat_template = open('./chat_template/chatml.jinja').read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        toker.chat_template = chat_template
        prompts = toker.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif 'vicuna' in model_option:
        chat_template = open('./chat_template/vicuna.jinja').read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        toker.chat_template = chat_template
        prompts = toker.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif 'llama2' in model_option:
        chat_template = open('./chat_template/llama-2-chat.jinja').read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        toker.chat_template = chat_template
        prompts = toker.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    request_outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    response = request_outputs[0].outputs[0].text
    return response
'''

def max_tokens_mapping(model_option):
    max_tokens_for_local_vllm_inference = {
        'qwen-1_8b-chat': 4096,
        'qwen-7b-chat': 4096,
        'qwen1.5-110b-chat': 4096,
    }
    return max_tokens_for_local_vllm_inference[model_option]

# Post-Process
def response_post_process(role_response):
    pattern = r'(?:(?:\[)?(?:Speaker|Listener|SPEAKER|LISTENER)(?:\])?(?:\:|：))(.*)'
    match = re.search(pattern, role_response)
    if match:
        return match.group(1).strip()
    return role_response.strip()

end_pattern = ['[END]', 'END.']

def end_trigger(response):
    for each_pattern in end_pattern:
        if each_pattern in response:
            return True
    return False

def delete_invalid_end(response, dialog_history, listener=False):
    index = 1 if not listener else 2

    def filter_end(string_pattern, response, dialog_history, index):
        if string_pattern in response and len(dialog_history) == index:
            response = response.replace(string_pattern, '')
        return response

    for each_pattern in end_pattern:
        response = filter_end(each_pattern, response, dialog_history, index)

    return response

def except_logger(dialog_history, cur_save_path):
    if len(dialog_history) == 1 or len(dialog_history) == 2:
        logging.info(f'Error Detected. PATH:{cur_save_path}, Index:{get_saved_number(cur_save_path)}')

def delete_valid_end_for_save(response):
    for each_pattern in end_pattern:
        response = response.replace(each_pattern, '').strip()
    return response

# Mimicker/Human Response
def opposite_response(dialog_history, response_option, args=None, model=None):
    if response_option == 'human':
        print('='*40)
        print()
        print(dialog_history[-1]['content'])
        print('='*40)
        response = input('Your input:')
    elif response_option == 'qwen':
        if args.qwen_api:
            response = qwen_return_answer(dialog_history)
        '''    
        else:
            # 本地vllm部署
            response = vllm_generate(messages=dialog_history, llm=model,
                                model_option=args.model, max_tokens=max_tokens_mapping(args.model),
                                temperature=args.temperature, top_p=args.top_p)
        '''
    else:
        response = gpt_return_answer(dialog_history)
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 激发情绪
    parser.add_argument('--emotion_induction', default='joy', choices=['joy', 'surprise', 'neutral', 'anger', 'sadness', 'disgust', 'fear'])
    # 基础情绪
    parser.add_argument('--base_emotion', default='neutral', choices=['joy', 'surprise', 'neutral', 'anger', 'sadness', 'disgust', 'fear'])
    # 生成对话段数 推荐 8 或者 16， 根据自己的时间来安排
    parser.add_argument('--generate_times', default=1, type=int)
    # vLLM本地部署需要的参数，不然不需要关注
    parser.add_argument('--tensor_parallel_size', default=1, type=int)
    parser.add_argument('--force_generate', action='store_true')
    parser.add_argument('--save_suffix', default='', type=str)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--temperature', type=float, default='0.6')
    parser.add_argument('--top_p', type=float, default='0.6')
    parser.add_argument('--gpu-memory-utilization', type=float, default='0.9')
    # 切换到listener进行激发
    parser.add_argument('--induce_as_listener', action='store_true')
    # 人工对话，测试激发人情绪的能力
    parser.add_argument('--human_input', action='store_true')
    # 使用Qwen API进行response
    parser.add_argument('--qwen_api', action='store_true')
    # 使用gpt还是qwen作为Mimicker
    parser.add_argument('--opposite_machine', default='gpt', choices=['gpt', 'qwen'])
    # 激发情绪的模型选择
    parser.add_argument('--model', default='gpt-3.5-turbo-0125',
                    choices=['gpt-3.5-turbo-0125', 'qwen-1_8b-chat', 'qwen-7b-chat', 'qwen1.5-110b-chat'])
    args = parser.parse_args()

    response_option = args.opposite_machine
    if args.human_input:
        response_option = 'human'

    induce_method = get_induce_prompt if 'qwen' not in args.model else get_induce_prompt_for_qwen

    # save dialog 相关setting
    speaker_role_save_prefix = 'speaker_role/'
    cur_save_path = f'./generated_dialogs_emotion/{speaker_role_save_prefix}generated_{args.base_emotion}_{args.emotion_induction}{args.save_suffix}.json'
    if args.induce_as_listener:
        listener_role_save_prefix = 'listener_role'
        cur_save_path = f'./generated_dialogs_emotion/{listener_role_save_prefix}/generated_{args.base_emotion}_{args.emotion_induction}{args.save_suffix}.json'
    temp = args.model.replace('-', '_')
    cur_save_path = cur_save_path.replace('./generated_dialogs_emotion', f'./generated_dialogs_emotion/{temp}')
    os.makedirs('/'.join(cur_save_path.split('/')[:-1]), exist_ok=True)
    generated_times = get_saved_number(cur_save_path)

    if args.force_generate:
        generated_times = 0

    # vllm本地部署
    model = None
    '''
    if 'gpt' not in args.model and generated_times < args.generate_times and not args.qwen_api:
        model = deploy_vllm_model_for_generation(models_for_local_vllm_inference[args.model],
                                            gpu_memory_utilization=args.gpu_memory_utilization,
                                            device=args.device,
                                            tensor_parallel_size=args.tensor_parallel_size)
    '''

    # 对话生成
    for generate_time in tqdm(range(generated_times, args.generate_times), position=0, leave=True):
        if args.induce_as_listener:
            listener_system_prompt = induce_method(args.emotion_induction) + ' ' + inducer_stop_prompt
            speaker_system_prompt = get_base_emotion_prompt(args.base_emotion) + ' ' + mimicker_stop_prompt

            begin_chat_prompt_suffix = f"""Please generate a topic to chat."""
            dialog_history_for_speaker = []
            dialog_history_for_listener = []
            dialog_history_for_speaker.append(get_message(speaker_system_prompt, 'system'))
            dialog_history_for_listener.append(get_message(listener_system_prompt, 'system'))

            for i in tqdm(range(8), position=1, leave=False):
                if i == 0:
                    temp_dialog_history = dialog_history_for_speaker + [get_message(begin_chat_prompt_suffix, 'user')]
                else:
                    temp_dialog_history = [dialog_history_for_speaker[0]] + [get_message(history_to_prompt(dialog_history_for_speaker, role='Speaker'), 'user')]

                response = opposite_response(temp_dialog_history, response_option, args=args, model=model)

                if i != 0:
                    response = response_post_process(response)
                response = delete_invalid_end(response, dialog_history_for_speaker)
                if end_trigger(response):
                    save_response = delete_valid_end_for_save(response)
                    if save_response:
                        dialog_history_for_speaker.append(get_message(save_response, 'user'))
                        dialog_history_for_listener.append(get_message(save_response, 'user'))
                    break
                dialog_history_for_speaker.append(get_message(response, 'user'))
                dialog_history_for_listener.append(get_message(response, 'user'))
                if i == 7:
                    break
                if 'gpt' not in args.model:
                    if args.qwen_api:
                        response = qwen_return_answer(temp_dialog_history)
                    '''
                    else:
                        response = vllm_generate(messages=dialog_history_for_listener, llm=model,
                                    model_option=args.model, max_tokens=max_tokens_mapping(args.model),
                                    temperature=args.temperature, top_p=args.top_p)
                    '''
                else:
                    response = gpt_return_answer(dialog_history_for_listener)
                response = delete_invalid_end(response, dialog_history_for_speaker, listener=True)
                if end_trigger(response):
                    save_response = delete_valid_end_for_save(response)
                    if save_response:
                        dialog_history_for_listener.append(get_message(save_response, 'assistant'))
                        dialog_history_for_speaker.append(get_message(save_response, 'assistant'))
                    break
                dialog_history_for_listener.append(get_message(response, 'assistant'))
                dialog_history_for_speaker.append(get_message(response, 'assistant'))

            except_logger(dialog_history_for_speaker, cur_save_path)
            dialog_history = [{'index': get_saved_number(cur_save_path), 'history': dialog_history_for_speaker}]
            dialog_save(dialog_history, save_path=cur_save_path)
            continue

        system_prompt_all = induce_method(args.emotion_induction) + ' ' + inducer_stop_prompt
        listener_system_prompt = get_base_emotion_prompt(args.base_emotion) + ' ' + mimicker_stop_prompt

        system_prompt_all_listener = listener_system_prompt
        begin_chat_prompt_suffix = f"""{induce_method(args.emotion_induction)} Please generate a topic that can help you to realize the goal. You are only need to speak your turn. """

        dialog_history = []
        dialog_history_for_listener = []
        dialog_history.append(get_message(system_prompt_all, 'system'))
        dialog_history_for_listener.append(get_message(system_prompt_all_listener, 'system'))

        for i in tqdm(range(8), position=1, leave=False):
            if i == 0:
                temp_dialog_history = dialog_history + [get_message(begin_chat_prompt_suffix, 'user')]
            else:
                temp_dialog_history = [dialog_history[0]] + [get_message(history_to_prompt(dialog_history, role='Speaker'), 'user')]
            if 'gpt' not in args.model:
                if args.qwen_api:
                    response = qwen_return_answer(temp_dialog_history)
                '''
                else:
                    response = vllm_generate(messages=temp_dialog_history, llm=model,
                                    model_option=args.model, max_tokens=max_tokens_mapping(args.model),
                                    temperature=args.temperature, top_p=args.top_p)
                '''
            else:
                response = gpt_return_answer(temp_dialog_history)
            if i != 0:
                response = response_post_process(response)
            response = delete_invalid_end(response, dialog_history_for_listener)
            if end_trigger(response):
                save_response = delete_valid_end_for_save(response)
                if save_response:
                    dialog_history.append(get_message(save_response, 'user'))
                    dialog_history_for_listener.append(get_message(save_response, 'user'))
                break

            dialog_history.append(get_message(response, 'user'))
            dialog_history_for_listener.append(get_message(response, 'user'))

            response = opposite_response(dialog_history_for_listener, response_option, args=args, model=model)

            response = delete_invalid_end(response, dialog_history_for_listener, listener=True)
            if end_trigger(response):
                save_response = delete_valid_end_for_save(response)
                if save_response:
                    dialog_history_for_listener.append(get_message(save_response, 'assistant'))
                    dialog_history.append(get_message(save_response, 'assistant'))
                break
            dialog_history_for_listener.append(get_message(response, 'assistant'))
            dialog_history.append(get_message(response, 'assistant'))

        except_logger(dialog_history_for_listener, cur_save_path)
        dialog_history = [{'index': get_saved_number(cur_save_path), 'history': dialog_history_for_listener}]
        dialog_save(dialog_history, save_path=cur_save_path)