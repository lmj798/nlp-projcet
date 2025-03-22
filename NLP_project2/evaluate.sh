python evaluate.py --evaluate_all --load_suffix 'qwen' \
    --result_save_path ./evaluate_results/qwen_7b_chat/comb_speaker --evaluate_comb \
    --save_suffix qwen_7b_chat --evaluate_multi_topk

python evaluate.py --evaluate_all --load_suffix 'qwen' \
    --result_save_path ./evaluate_results/qwen_7b_chat/comb_listener --evaluate_comb \
    --save_suffix qwen_7b_chat --evaluate_listener --evaluate_multi_topk