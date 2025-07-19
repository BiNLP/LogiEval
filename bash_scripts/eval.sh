DATASET='logiqa2'

python evaluate.py \
    --model_name "/data/home/scyb224/Workspace/LLMs/Qwen2.5-1.5B" \
    --datasets ${DATASET} \
    --batch_size 64 \
    --temperature 0.6 \
    --max_tokens 4096 \
    --top_p 0.95 \
    --sampling_method "direct" \
    --num_samples 5 \
    --use_vllm \
    --save_intermediate \
    --tensor_parallel_size 1 \
    --output_dir "./results/${DATASET}" 