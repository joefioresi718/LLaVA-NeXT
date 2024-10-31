import torch.utils
from PIL import Image
import copy
import json
import time
import torch
from torch.utils.data import DataLoader
# from transformers import AutoProcessor, AutoModelForCausalLM, LlavaOnevisionForConditionalGeneration, MllamaForConditionalGeneration, Qwen2VLForConditionalGeneration, GenerationConfig
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaOnevisionForConditionalGeneration, MllamaForConditionalGeneration, Qwen2VLForConditionalGeneration, GenerationConfig
from transformers import GenerationConfig
import os
import gc
import pandas as pd
import av
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def read_video_pyav(video_path, num_frames):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    use_flash_attention_2=True,
    device_map='auto'
)

model.eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto', device_map='auto')
tokenizer = processor.tokenizer if 'InternVL' not in model_id else processor
choices_ids = ['A', 'B', 'C', 'D']
choice_token_ids = [tokenizer.encode(choice_id, add_special_tokens=False)[0] for choice_id in choices_ids]

dataset_path = '../Video-LLaVA/V-MMVP_ft'
dataset = pd.read_csv(f'{dataset_path}/V-MMVP_ft_final.csv')

num_frames = 16

all_results = []
total_correct, total_count = 0, 0
results = []
with torch.inference_mode():
    for i, row in dataset.iterrows():
        if row['dataset'] == 'kinetics400':
            break

        total_count += 1

        mc_prompt = row['question'] + ' Choose the best option from the following that answers the question:\n' + row['options'].split('(b)')[0] + '\n(b)' + row['options'].split('(b)')[1] + '\nBest option: ('

        # System Message
        # You are a helpful assistant that can answer question for an image. I will provide you 4 options.
        # Response Format
        # Choice: A single character from A, B, C, D.
        # mc_prompt = annotations[0]['question'] + '\n' + ' '.join(annotations[0]['choices']) + '\n' + 'Choice: Return only a single character from A, B, C, D.'
        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can answer question for an image. I will provide you 4 options."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": mc_prompt},
                    ],
                },
        ]
        if 'llama' in model_id:
            # conversation[0]['role'] = 'user'
            conversation = [conversation[1]]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        video_paths = [read_video_pyav(os.path.join(dataset_path, row['pair_path'], vid), num_frames) for vid in [row['video1'], row['video2']]]



        inputs1 = processor(images=video_paths[0], text=prompt, return_tensors='pt').to(model.device, torch.bfloat16)
        inputs2 = processor(images=video_paths[1], text=prompt, return_tensors='pt').to(model.device, torch.bfloat16)
        output1 = model.generate(**inputs1, max_new_tokens=1, do_sample=False, output_logits=True, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True)
        output2 = model.generate(**inputs2, max_new_tokens=1, do_sample=False, output_logits=True, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True)

        logits1 = output1['logits'][0]
        log_probs1 = torch.log_softmax(logits1, dim=-1)
        log_likelihoods1 = [log_probs1[0, choice_token_id].item() for choice_token_id in choice_token_ids]

        # Select the answer with the highest log likelihood
        best_choice_idx1 = torch.argmax(torch.tensor(log_likelihoods1))
        pred_answer1 = choices_ids[best_choice_idx1]

        logits2 = output2['logits'][0]
        log_probs2 = torch.log_softmax(logits2, dim=-1)
        log_likelihoods2 = [log_probs2[0, choice_token_id].item() for choice_token_id in choice_token_ids]

        # Select the answer with the highest log likelihood
        best_choice_idx2 = torch.argmax(torch.tensor(log_likelihoods2))
        pred_answer2 = choices_ids[best_choice_idx2]

        correct_answer1 = row['v1_correct_answer']
        correct_answer2 = row['v2_correct_answer']

        correct1 = pred_answer1 == correct_answer1
        correct2 = pred_answer2 == correct_answer2

        print(pred_answer1, correct_answer1, correct1)
        print(pred_answer2, correct_answer2, correct2)
        if correct1 and correct2:
            total_correct += 1



        torch.cuda.empty_cache()
        gc.collect()
        # time.sleep(5)


print(f"Total correct: {total_correct}/{total_count}")
print(f"Accuracy: {total_correct/total_count:.2f}")

# Save results
# results_dir = 'results'
# os.makedirs(results_dir, exist_ok=True)
# results_file = os.path.join(results_dir, f'mmspubench_mllama-11b.json')

# with open(results_file, 'w') as f:
#     json.dump(results, f, indent=4)