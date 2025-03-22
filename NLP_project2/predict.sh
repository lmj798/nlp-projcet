# listener
for base_emotion in joy anger sadness #surprise fear disgust neutral
do
    for emotion in joy anger sadness #surprise fear disgust neutral
    do
        if [ "$emotion" == "$base_emotion" ]; then
            continue
        else
            python ./predict.py --emotion_induction $emotion --evaluate_start_point 0 --evaluate_all --load_suffix 'qwen' \
                --base_emotion $base_emotion --comb_evaluate --model_prefix_save 'qwen_7b_chat' --device cuda:0 --evaluate_listener  --inplace
        fi
    done
done

# speaker
for base_emotion in joy anger sadness #surprise fear disgust neutral
do
    for emotion in joy anger sadness #surprise fear disgust neutral
    do
        if [ "$emotion" == "$base_emotion" ]; then
            continue
        else
            # inplace : 原地替换上次的检验结果，不设置的话就是增量检验。
            python ./predict.py --emotion_induction $emotion --evaluate_start_point 0 --evaluate_all --load_suffix 'qwen' \
                --base_emotion $base_emotion --comb_evaluate --model_prefix_save 'qwen_7b_chat' --device cuda:0 --inplace
        fi
    done
done