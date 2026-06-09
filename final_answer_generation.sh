# Correct 1K Ours
CUDA_VISIBLE_DEVICES="4,5,6,7" python final_answer.py --folder_path "/workspace/daeyong/inference_results/dev_kg_correct_1ksample_no_premature_conclusion_10_3_qwen3_8b_no_premature_conclusion"
CUDA_VISIBLE_DEVICES="4,5,6,7" python oss_answer_binary.py --folder_path "/workspace/daeyong/inference_results/dev_kg_correct_1ksample_no_premature_conclusion_10_3_qwen3_8b_no_premature_conclusion"

# Wrong Ours
# CUDA_VISIBLE_DEVICES="4,5,6,7" python final_answer.py --folder_path "/workspace/daeyong/inference_results/dev_kg_wrong_2wiki_ver3_newprompt_v2_qwen2.5_7b_2wiki_added_ver3_checkpoint_200"
# CUDA_VISIBLE_DEVICES="4,5,6,7" python oss_answer_binary.py --folder_path "/workspace/daeyong/inference_results/dev_kg_wrong_2wiki_ver3_newprompt_v2_qwen2.5_7b_2wiki_added_ver3_checkpoint_200"


# Self-feedback
# CUDA_VISIBLE_DEVICES="4,5,6,7" python final_answer_self_feedback.py --folder_path /workspace/daeyong/inference_results/self_feedback_Meta-Llama-3.1-8B-Instruct
# CUDA_VISIBLE_DEVICES="4,5,6,7" python final_answer_self_feedback.py --folder_path /workspace/daeyong/inference_results/self_feedback_gemma-3-12b-it
# CUDA_VISIBLE_DEVICES="4,5,6,7" python oss_answer_binary_self_feedback.py --folder_path /workspace/daeyong/inference_results/self_feedback_Meta-Llama-3.1-8B-Instruct
# CUDA_VISIBLE_DEVICES="4,5,6,7" python oss_answer_binary_self_feedback.py --folder_path /workspace/daeyong/inference_results/self_feedback_gemma-3-12b-it
