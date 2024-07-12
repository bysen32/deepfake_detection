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
import torch
from multiprocessing import Pool, Queue, Process
from functools import partial


face_detector = dlib.get_frontal_face_detector()
predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)


def facecrop(frame_path):

    land_path = frame_path.replace("/frames", "/landmarks").replace(".jpg", ".npy")
    # if os.path.isfile(land_path):
    #     return

    try:
        frame = cv2.imread(frame_path)
        faces = face_detector(frame, 1)
        if len(faces) == 0:
            tqdm.write('No faces in {}'.format(frame_path))
            return

        landmarks = []
        size_list = []
        for face in faces:
            landmark = face_predictor(frame, face)
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1-x0) * (y1-y0)
            size_list.append(face_s)
            landmarks.append(landmark)

            # x0 = face.left()
            # y0 = face.top()
            # x1 = face.right()
            # y1 = face.bottom()
            # frame = cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        landmarks = np.concatenate(landmarks).reshape( (len(size_list),) + landmark.shape ) # reshape according size_list
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

        # save landmark file
        land_folder = os.path.dirname(land_path)
        os.makedirs(land_folder, exist_ok=True)
        np.save(land_path, landmarks)

        # save_path = frame_path.replace("/frames", "/temp").replace(".jpg", "_face.jpg")
        # save_dir = os.path.dirname(save_path)
        # os.makedirs(save_dir, exist_ok=True)
        # cv2.imwrite(save_path, frame)

    except Exception as e:
        print(e)
        print(f"Error in {frame_path}")


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', choices=['multiFFDI-phase1',])
    parser.add_argument('-p', dest='phase', choices=['train', 'val'])
    args = parser.parse_args()

    if args.dataset == 'multiFFDI-phase1':
        dataset_path = f'data/{args.dataset}/{args.phase}set'
    else:
        raise NotImplementedError

    device = torch.device('cuda')

    frame_root = os.path.join(dataset_path, 'frames')
    landmark_root = os.path.join(dataset_path, 'landmarks')

    search = os.path.join(frame_root, '*.jpg')
    frame_paths = sorted(glob(search, recursive=True)) 
    frame_paths = frame_paths[:10]
    print(f"{len(frame_paths)} : frames are exist in {args.dataset}/{args.phase}")

    search = os.path.join(landmark_root, '*.npy')
    landmark_paths = sorted(glob(search, recursive=True))
    landmark_count = len(landmark_paths)
    # frame_paths = frame_paths[landmark_count:]

    with Pool(processes=os.cpu_count()-8) as p:
        with tqdm(total=len(frame_paths)) as pbar:
            for it in p.imap_unordered(partial(facecrop), frame_paths):
                pbar.update()
