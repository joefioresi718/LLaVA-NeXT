# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import pandas as pd
import os.path
import copy
import torch
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames, frame_time, video_time


pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
max_frames_num = 64

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, load_4bit=False, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
# model = model.to(torch.bfloat16)

dataset_path = '../Video-LLaVA/V-MMVP_ft'
dataset = pd.read_csv(f'{dataset_path}/V-MMVP_ft_final.csv')

all_results = []
for i, row in dataset.iterrows():
    if row['dataset'] == 'kinetics400':
        break
    
    video_paths = [os.path.join(dataset_path, row['pair_path'], vid) for vid in [row['video1'], row['video2']]]
    video1, frame_time1, video_time1 = load_video(video_paths[0], max_frames_num, 1, force_sample=True)
    video2, frame_time2, video_time2 = load_video(video_paths[1], max_frames_num, 1, force_sample=True)
    video1 = image_processor.preprocess(video1, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video2 = image_processor.preprocess(video2, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video1 = [video1]
    video2 = [video2]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    # time_instruciton1 = f"The video lasts for {video_time1:.2f} seconds, and {len(video1[0])} frames are uniformly sampled from it. These frames are located at {frame_time1}. Please answer the following questions related to this video."
    # question1 = DEFAULT_IMAGE_TOKEN + f"{time_instruciton1}\nPlease describe this video in detail."
    question = DEFAULT_IMAGE_TOKEN + row['question'] + ' Choose the best option from the following that answers the question:\n' + row['options'].split('(b)')[0] + '\n(b)' + row['options'].split('(b)')[1] + '\nBest option: ('
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # with torch.inference_mode():
    cont1 = model.generate(
        input_ids,
        images=video1,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    cont2 = model.generate(
        input_ids,
        images=video2,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )

    outputs1 = tokenizer.batch_decode(cont1, skip_special_tokens=True)[0].strip()
    outputs2 = tokenizer.batch_decode(cont2, skip_special_tokens=True)[0].strip()
    print(outputs1)
    print(outputs2)

    result = {
        'dataset': row['dataset'],
        'pair_path': row['pair_path'],
        'video1': row['video1'],
        'video2': row['video2'],
        'question': row['question'],
        'options': row['options'],
        'answer1': outputs1,
        'answer2': outputs2,
        'v1_correct': row['v1_correct_answer'],
        'v2_correct': row['v2_correct_answer'],
        'clip_similarity': row['clip_similarity'],
        'vssl_similarity': row['vssl_similarity']
    }

    all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df.to_csv(f'{dataset_path}/V-MMVP_ft_llavavideo.csv', index=False)
