from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
import dlib
from imutils import face_utils
import multiprocessing
from functools import partial



def extract_frames(
    video_path,  # 视频文件路径
    num_frames=10
):
    cap_org = cv2.VideoCapture(video_path)
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, frame_count_org - 1,
                             num_frames, endpoint=True, dtype=int)
    frame_width = int(cap_org.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_org.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while frame_width >= 1024 or frame_height >= 1024:
        frame_width = frame_width // 2
        frame_height = frame_height // 2

    print(video_path)
    if '/QiYuan/' in video_path:
        frame_folder = video_path.replace('/videos/', '/frames/')
    else:
        frame_folder = video_path.replace('/videos/', '/frames/').replace('.mp4', '/')

    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        if not ret_org:
            tqdm.write('Frame read {} Error! : {}'.format(
                cnt_frame, os.path.basename(video_path)))
            break
        if cnt_frame not in frame_idxs:
            continue

        # save frame file
        image_path = os.path.join(frame_folder, f"{cnt_frame:03d}.png")
        os.makedirs(frame_folder, exist_ok=True)
        frame_org = cv2.resize(frame_org, (frame_width, frame_height))

        cv2.imwrite(image_path, frame_org)

    cap_org.release()
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', choices=['multiDFFV-phase1'])
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    args = parser.parse_args()

    if args.dataset == 'Original':
        dataset_path = f'data/FaceForensics++/original_sequences/youtube/{args.comp}/'
    elif args.dataset == 'DeepFakeDetection_original':
        dataset_path = f'data/FaceForensics++/original_sequences/actors/{args.comp}/'
    elif args.dataset in ['DeepFakeDetection', 'FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_path = f'data/FaceForensics++/manipulated_sequences/{args.dataset}/{args.comp}/'
    elif args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
        dataset_path = f'data/Celeb-DF-v1/{args.dataset}/'
    elif args.dataset == 'DFDCP_Original':
        dataset_path = f'data/DFDCP/original_videos/'
    elif args.dataset == 'DFDCP_MethodA':
        dataset_path = f'data/DFDCP/method_A/'
    elif args.dataset == 'DFDCP_MethodB':
        dataset_path = f'data/DFDCP/method_B/'
    elif args.dataset in ['DFDC']:
        dataset_path = 'data/DFDC/'
    elif args.dataset == 'QiYuan':
        dataset_path = 'data/QiYuan/'
    else:
        raise NotImplementedError

    video_root = dataset_path + 'videos/'
    print(video_root)
    video_paths = sorted(glob(video_root + '**/*.avi', recursive=True)) + \
                    sorted(glob(video_root + '**/*.mp4', recursive=True)) + \
                    sorted(glob(video_root + '**/*.flv', recursive=True))
    video_paths = sorted(video_paths)
    video_paths = video_paths[2600:]
    # 特殊处理 error_list = ['011埃马纽埃尔·马克龙', '039约翰·特拉沃塔']
    # video_paths = [video_path for video_path in video_paths if '011埃马纽埃尔·马克龙' in video_path or '039约翰·特拉沃塔' in video_path]

    print(f"{len(video_paths)} : videos are exist in {args.dataset}")

    with multiprocessing.Pool(os.cpu_count()//3) as p:
        with tqdm(total=len(video_paths)) as pbar:
            pfunc = partial(extract_frames, num_frames=args.num_frames)
            for _ in p.imap(pfunc, video_paths):
                pbar.update()
