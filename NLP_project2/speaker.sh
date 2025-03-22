count=0
for base_emotion in joy anger sadness #surprise fear disgust neutral
do
    for emotion in joy anger sadness #surprise fear disgust neutral
    do
        echo "Counter: $count; base emotion: $base_emotion; induce emotion: $emotion"
        if [ "$emotion" == "$base_emotion" ]; then
            count=$((count + 1))
            continue
        else
            # 如果不使用API，采用vllm本地部署的话，请根据自己的显存资源设置 gpu-memory-utilization
            # --opposite_machine 设置 machine进行模拟
            # --human_input 自己进行对话测试
            # --induce_as_listener 切换到listener端做情感激发
            # --generate_times 生成对话的条数
            CUDA_VISIBLE_DEVICES=0 python ./main.py --emotion_induction $emotion --generate_times 16 \
                --base_emotion $base_emotion --model 'qwen-7b-chat' --save_suffix 'qwen' --device "0" \
                --tensor_parallel_size 1 --gpu-memory-utilization 0.3 --opposite_machine qwen --qwen_api
        fi
        count=$((count + 1))
    done
done