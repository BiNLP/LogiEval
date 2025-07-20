DATASET='logiqa'

python evaluate.py \
    --model_name "/home/chenzhb/Workspaces/LLMs/ReasonFlux-PRM-Qwen-2.5-7B" \
    --datasets ${DATASET} \
    --batch_size 64 \
    --temperature 0.6 \
    --max_tokens 4096 \
    --top_p 0.95 \
    --sampling_method "direct" \
    --num_samples 5 \
    --use_vllm \
    --save_intermediate \
    --tensor_parallel_size 2 \
    --output_dir "./results/${DATASET}" 