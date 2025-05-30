import math
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import *
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from torch.nn.functional import cosine_similarity

from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import requests
from PIL import Image
from io import BytesIO
import re
import pickle
import torchvision.transforms as T


import numpy as np
from torch.utils.data import Dataset

def split_dataset(dataset, n):
    """Split a Dataset into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(dataset) / n)
    chunks = []
    
    for i in range(0, len(dataset), chunk_size):
        chunk = [dataset[j] for j in range(i, min(i + chunk_size, len(dataset)))]
        chunks.append(chunk)
    
    return chunks

def get_chunk(dataset, n, k):
    chunks = split_dataset(dataset, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--model_name', help='', default='llava')
    parser.add_argument('--data_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--conv_mode", type=str, default='v1')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()
def load_video(vis_path, n_clips=1, num_frm=None):
    """
    Load video frames from a video file or image files from a directory.

    Parameters:
    vis_path (str): Path to the video file or directory.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames or images.
    """
    
    # Check if vis_path is a directory or a video file
    if os.path.isdir(vis_path):
        # Load images from directory
        image_files = sorted([os.path.join(vis_path, f) for f in os.listdir(vis_path) if f.endswith(('png', 'jpg', 'jpeg'))])
        total_num_frm = len(image_files)
        clip_imgs = [Image.open(img_path) for img_path in image_files]
    else:
        # Load video with VideoReader
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        # Currently, this function supports only 1 clip
        assert n_clips == 1

        if num_frm is None:
            fps = vr.get_avg_fps()
            num_frm = int(total_frame_num // fps)
            num_frm=min(num_frm,1000)

        # Calculate total number of frames to extract
        total_num_frm = min(total_frame_num, num_frm)
        # Get indices of frames to extract
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
        
        # Extract frames as numpy array
        try:
            img_array = vr.get_batch(frame_idx).asnumpy()
        except:
            img_array = vr.get_batch([0]).asnumpy()
            total_num_frm = 1
        
        # Set target image height and width
        target_h, target_w = 336, 336
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

        # Reshape array to match number of clips and frames
        img_array = img_array.reshape(
            (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
        # Convert numpy arrays to PIL Image objects
        clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


def split_image(image, n=2):
    if n==1: return [image]
    width, height = image.size
    block_width = width // n
    block_height = height // n

    blocks = []

    for i in range(n):
        for j in range(n):
            left = j * block_width
            upper = i * block_height
            right = (j + 1) * block_width
            lower = (i + 1) * block_height
            block = image.crop((left, upper, right, lower))
            blocks.append(block)
    blocks.append(image)

    return blocks



def get_model_output(model, video_processor, tokenizer, video, qs, args):
    num_frm=None

    qs+="\nAnswer with one letter."

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()#+"\nOnly give the best option."+"Best option:("

    try:
        video_frames = load_video(video,num_frm=num_frm)
    except:
        return "N/A"
    temporal_len=len(video_frames)
    N=getattr(model.config,'resolution_ratio', 1)
    images=[]
    for video_frame in video_frames:
        images.extend(split_image(video_frame,n=N))

    image_tensor = video_processor.preprocess(images, return_tensors='pt')['pixel_values']

    image_tensor = image_tensor.to(model.device, dtype=torch.float16).unsqueeze(0)

    bsz,N2_x_temporal,rgb,height,width=image_tensor.size()
    image_tensor=image_tensor.view(bsz,temporal_len,-1,rgb,height,width)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(f"################################## image tensor:{image_tensor.shape} ############################")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, "json/"+v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type'],
            'candidates':self.data_list[idx]['data']['candidates']
        }

def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[index3-1:index3]

    if pred==gt_option:
        flag=True

    return flag

def first_char_as_answer(res, n=4):
    options = [chr(ord('A') + i) for i in range(n)]
    for s in res:
        if s in options:
            return s
    return options[0]

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """

    data_list = {
        "Action Sequence": ("action_sequence.json", f"{args.data_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", f"{args.data_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", f"{args.data_dir}/video/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", f"{args.data_dir}/video/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", f"{args.data_dir}/video/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", f"{args.data_dir}/video/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", f"{args.data_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", f"{args.data_dir}/video/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", f"{args.data_dir}/video/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", f"{args.data_dir}/video/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", f"{args.data_dir}/video/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", f"{args.data_dir}/video/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", f"{args.data_dir}/video/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", f"{args.data_dir}/video/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", f"{args.data_dir}/video/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", f"{args.data_dir}/video/nturgbd/", "video", False),
        "Character Order": ("character_order.json", f"{args.data_dir}/video/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", f"{args.data_dir}/video/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", f"{args.data_dir}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", f"{args.data_dir}/video/clevrer/video_validation/", "video", False),
    }

   

    data_dir = args.data_dir
    save_path = os.path.join(args.output_dir, f"save_{args.output_name}.json")
    
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")
    result_path= os.path.join(args.output_dir, f"result_{args.output_name}.json")

    dataset = MVBench_dataset(data_dir, data_list)
    chunk_dataset=get_chunk(dataset, args.num_chunks, args.chunk_idx)

    # Initialize the model
    # model_name = get_model_name_from_path(args.model_path)
    # model_name='llava'
    model_name=args.model_name
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)



    


    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    for example in tqdm(chunk_dataset):
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        video_path=example["video"]
        question=example["question"]

        pred = get_model_output(model, processor, tokenizer, video_path, question, args)
       
        gt = example['answer']
        ans=first_char_as_answer(pred,len(example["candidates"]))
        res_dict={
            'pred': pred,
            'gt': gt,
            'question':example['question'],
            'question_type':example['task_type'],
            'video':example['video'],
            'ans': ans,
            'acc': 1 if ans==first_char_as_answer(gt,len(example["candidates"])) else 0
        }


        ans_file.write(json.dumps(res_dict) + "\n")
        res_list.append(res_dict)
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
            
    ans_file.close()
    with open(f"{save_path}.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    total=0
    idx=0
    for k, v in acc_dict.items():
        idx+=1
        final_res[k] = v[0] / v[1] * 100  
        total+=final_res[k]
    final_res['Avg'] = total /idx 
    print(final_res)

    with open(result_path, "w") as f:
        json.dump(final_res, f)

if __name__ == "__main__":
    # args = parse_args()

    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--model_name', help='', default='llava')
    parser.add_argument('--data_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--conv_mode", type=str, default='v1')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    args = parser.parse_args()

    run_inference(args)
