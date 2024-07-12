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


def facecrop(
    model,
    video_path,
    frame_folder,
    period=1,
    num_frames=10
):
    # valid_id = ['002', '006', '009', '027', '050', '059', '060', '088']
    # is_valid = False
    # for id in valid_id:
    #     if id in frame_folder:
    #         is_valid = True
    # if not is_valid:
    #     return

    frame_paths = sorted(glob(frame_folder + '*.png'))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        landmark_path = frame_path.replace("/frames/", "/landmarks/").replace(".png", ".npy")
        if not os.path.isfile(landmark_path):
            continue
        # faces = model.predict_jsons(frame)
        # try:
        #     if len(faces) == 0:
        #         print(faces)
        #         tqdm.write('No faces in {}:{}'.format(
        #             frame_path, os.path.basename(video_path)))
        #         continue
        #     face_s_max = -1
        #     landmarks = []
        #     size_list = []
        #     for face_idx in range(len(faces)):
        #         x0, y0, x1, y1 = faces[face_idx]['bbox']
        #         landmark = np.array([[x0, y0], [x1, y1]] + faces[face_idx]['landmarks'])
        #         face_s = (x1-x0) * (y1-y0)
        #         size_list.append(face_s)
        #         landmarks.append(landmark)
        # except Exception as e:
        #     print(f'error in {frame_path}:{video_path}')
        #     print(e)
        #     continue

        # landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
        # landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

        # land_path = frame_path.replace("/frames", "/retina").replace(".png", ".npy")
        # os.makedirs(os.path.dirname(land_path), exist_ok=True)
        # np.save(land_path, landmarks)
        crop_path = frame_path.replace("/frames/", "/crops/")
        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
        landmark = np.load(landmark_path)[0]
        x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
        x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
        width = x1 - x0
        height = y1 - y0
        
        try:
            crop_face = frame[int(y0-0.2*height):int(y1+0.2*height), int(x0-0.2*width):int(x1+0.2*width)]
            crop_face = cv2.resize(crop_face, (224, 224))
            cv2.imwrite(crop_path, crop_face)
        except Exception as e:
            print(f'error in {frame_path}:{video_path}')
            print(e)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', choices=['DeepFakeDetection_original',
                                                       'DeepFakeDetection',
                                                       'FaceShifter',
                                                       'Face2Face',
                                                       'Deepfakes',
                                                       'FaceSwap',
                                                       'NeuralTextures',
                                                       'Original',
                                                       'Celeb-real',
                                                       'Celeb-synthesis',
                                                       'YouTube-real',
                                                       'DFDC',
                                                       'DFDCP',
                                                       'DFDCP_MethodA',
                                                       'DFDCP_MethodB',
                                                       'DFDCP_Original',
                                                       ])
    parser.add_argument('-c', dest='comp',
                        choices=['raw', 'c23', 'c40'], default='raw')
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    args = parser.parse_args()

    need2crop = [
        # 'DeepFakeDetection_original',
        # 'DeepFakeDetection',
        # 'FaceShifter',
        # 'Face2Face',
        # 'Deepfakes',
        # 'FaceSwap',
        # 'NeuralTextures',
        # 'Original',
        # 'Celeb-real',
        # 'Celeb-synthesis',
        # 'YouTube-real',
        'DFDC',
    ]
    for dataset in need2crop:
        # dataset
        args.dataset = dataset
        args.comp = 'c23'

        if args.dataset == 'Original':
            dataset_path = f'data/FaceForensics++/original_sequences/youtube/{args.comp}/'
        elif args.dataset == 'DeepFakeDetection_original':
            dataset_path = f'data/FaceForensics++/original_sequences/actors/{args.comp}/'
        elif args.dataset in ['DeepFakeDetection', 'FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
            dataset_path = f'data/FaceForensics++/manipulated_sequences/{args.dataset}/{args.comp}/'
        elif args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
            dataset_path = f'data/Celeb-DF-v1/{args.dataset}/'
        elif args.dataset in ['DFDC']:
            dataset_path = f'data/{args.dataset}/'
        elif args.dataset == 'DFDCP_Original':
            dataset_path = f'data/DFDCP/original_videos/'
        elif args.dataset == 'DFDCP_MethodA':
            dataset_path = f'data/DFDCP/method_A/'
        elif args.dataset == 'DFDCP_MethodB':
            dataset_path = f'data/DFDCP/method_B/'
        else:
            raise NotImplementedError

        device = torch.device('cuda')

        model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
        model.eval()

        movies_path = dataset_path + 'videos/'
        movies_path_list = sorted(glob(movies_path + '**/*.mp4', recursive=True)) + \
                            sorted(glob(movies_path + '**/*.avi', recursive=True))

        # print("{} : videos are exist in {}".format(
        #     len(movies_path_list), args.dataset))

        n_sample = len(movies_path_list)

        for i in tqdm(range(n_sample)):
            video_path = movies_path_list[i]
            if '/QiYuan/' in video_path:
                frame_folder = video_path.replace('/videos/', '/frames/')
            else:
                frame_folder = video_path.replace('/videos/', '/frames/').replace('.mp4', '/')
            facecrop(model, video_path, frame_folder=frame_folder, num_frames=args.num_frames)
