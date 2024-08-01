import os, sys
import numpy as np
import cv2
import torch

from .predict_single import Model



def load_retinaface_model(filename='retinaface_resnet50_2020-07-20_old_torch.pth', device='cuda'):
    # detector = Model(max_size=512, device=device)
    detector = Model(max_size=112, device=device)
    state_dict = torch.load(filename, map_location=device)
    detector.load_state_dict(state_dict)
    detector.eval()
    return detector


def save_5landmarks_textfile(landmarks=[[]], file_path='name.txt'):
    if len(landmarks) != 5:
        raise ValueError("The list of landmarks must contain exactly 5 items.")
    
    with open(file_path, 'w') as file:
        for landmark in landmarks:
            if len(landmark) != 2:
                raise ValueError("Each landmark must contain exactly 2 coordinates.")
            file.write(f"{landmark[0]} {landmark[1]}\n")


def detect_5p(img_folder, detector=Model(), output_folder='detections', draw_landmarks=False):
    detector.eval()

    img_path = img_folder
    names = [i for i in sorted(os.listdir(img_path)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]
    # print('names:', names)

    output_folder = os.path.join(img_path, output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for i in range(0, len(names)):
        name = names[i]
        full_image_name = os.path.join(img_path, name)
        txt_name = '.'.join(name.split('.')[:-1]) + '.txt'
        full_txt_name = os.path.join(img_path, 'detections', txt_name) # 5 facial landmark path for each image
        
        if not os.path.isfile(full_txt_name):
            if i == 0:
                print('detecting 5 landmarks')

            print('%05d' % (i), ' ', name, ' ', full_txt_name)
            img_bgr = cv2.imread(full_image_name)
            rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = detector.predict_jsons(rgb_image)
            # print('results:', results)

            best_result = results[0]
            for result in results:
                if result['score'] > best_result['score']:
                    best_result = result
            landmarks5 = best_result['landmarks']
            save_5landmarks_textfile(landmarks5, full_txt_name)

        # sys.exit(0)