# 1. Import images
# 2. Compute avg. pixel values

import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def import_from_folder(folder_path):
    files = filter(lambda x: '.png' in x, os.listdir(folder_path))
    images = []
    for f in tqdm(files):
        path = os.path.join(folder_path, f)
        img = Image.open(path)
        images.append((f, np.mean(img, axis=(0, 1))))
    return images

def select_img(px, images):
    # Brute force method
    best_dist = 1e9
    for (f, avg_px) in images:
        dist = np.linalg.norm(px - avg_px)
        if dist < best_dist:
            best_f = f
            best_dist = dist
    return best_f


if __name__ == '__main__':
    images = import_from_folder("images")
    target_img = Image.open('images/0.png')
    target_img_np = np.asarray(target_img)
    print(target_img_np)
    new_img_np = np.zeros((32*32, 32*32, 3))
    for i in tqdm(range(32)):
        for j in range(32):
            alt_img_f = select_img(target_img_np[i, j], images)
            alg_img = Image.open('images/'+alt_img_f)
            alg_img_np = np.asarray(alg_img)
            new_img_np[i*32:(i+1)*32, j*32:(j+1)*32] = alg_img_np
    Image.fromarray(np.uint8(new_img_np)).save('result.png')
