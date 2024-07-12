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
from imutils import face_utils
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import torch

import multiprocessing
from functools import partial



def facecrop(frame_path, model):
    land_path = frame_path.replace("/frames/", "/retinas/").replace(".jpg", ".npy")
    # if os.path.exists(land_path): # 已有landmark
    #     return

    frame = cv2.imread(frame_path)
    faces = model.predict_jsons(frame)
    try:
        if len(faces) == 0:
            tqdm.write(f'No faces in {frame_path}')
            return

        landmarks = []
        size_list = []
        for face_idx in range(len(faces)):
            x0, y0, x1, y1 = faces[face_idx]['bbox']
            landmark = np.array([[x0, y0], [x1, y1]] + faces[face_idx]['landmarks'])
            face_s = (x1-x0) * (y1-y0)
            size_list.append(face_s)
            landmarks.append(landmark)

            frame = cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        cv2.imwrite(frame_path.replace("/frames/", "/temp/").replace(".jpg", "_face.jpg"), frame)

    except Exception as e:
        print(f'error in {frame_path}')
        print(e)
        return

    landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
    landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

    land_folder = os.path.dirname(land_path)
    os.makedirs(land_folder, exist_ok=True)
    np.save(land_path, landmarks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', choices=['multiFFDI-phase1',])
    parser.add_argument('-p', dest='phase', choices=['train', 'val'])
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    args = parser.parse_args()

    if args.dataset == 'multiFFDI-phase1':
        dataset_path = f'data/{args.dataset}/{args.phase}set'
    else:
        raise NotImplementedError
    
    frame_root = os.path.join(dataset_path, 'frames')
    retina_root = os.path.join(dataset_path, 'retinas')
    
    search = os.path.join(frame_root, '*.jpg')
    frame_paths = sorted(glob(search, recursive=True))
    frame_paths = frame_paths[:10]
    print(f"{len(frame_paths)} : frames are exist in dataset:{args.dataset}/{args.phase}")

    device = torch.device('cuda')
    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    # Set the start method to 'spawn'
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=12) as p:
        with tqdm(total=len(frame_paths)) as pbar:
            for _ in p.imap_unordered(partial(facecrop, model=model), frame_paths):
                pbar.update()

    # for frame_path in tqdm(frame_paths):
    #     facecrop(frame_path)

