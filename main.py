# 1. Import images
# 2. Compute avg. pixel values

import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pickle
import random

def import_from_folder(folder_path, save_pkl=True):    
    files = filter(lambda x: '.png' in x, os.listdir(folder_path))
    images = []
    for f in tqdm(files):
        path = os.path.join(folder_path, f)
        img = Image.open(path)
        images.append((f, np.mean(img, axis=(0, 1))))
    
    if save_pkl:
        with open('avg_px.pkl', 'wb') as f:
            pickle.dump(images, f)

    return images

def select_img_km(px, images, centers, groups, remove_used=False, rand=False):
    #k-means
    best_dist = 1e9
    for ind, c in enumerate(centers):
        dist = np.linalg.norm(px - c)
        if dist < best_dist and len(groups[ind]) > 0:
            best_dist = dist
            best_ind = ind
    if rand:
        best_f = random.choice(list(groups[best_ind].keys()))
    else:
        best_dist = 1e9
        for f in groups[best_ind].keys():
            dist = np.linalg.norm(px - groups[best_ind][f])
            if dist < best_dist:
                best_dist = dist
                best_f = f
    if remove_used:
        del groups[best_ind][best_f]
    return best_f

def select_img(px, images, remove_used=False):
    # Brute force method
    best_dist = 1e9
    for (f, avg_px) in images:
        dist = np.linalg.norm(px - avg_px)
        if dist < best_dist:
            best_f = f
            best_dist = dist
            best_avg_px = avg_px
    if remove_used:
        images.remove((best_f, best_avg_px))
    return best_f

def k_means(images, k=20, num_iters=10, save_pkl=True):
    centers = [(random.randint(0,255), random.randint(0,255), \
                random.randint(0,255)) for _ in range(k)]
    groups = [{} for _ in range(k)]

    for _ in tqdm(range(num_iters)):
        for (f, avg_px) in images:
            best_dist = 1e9
            for i, c in enumerate(centers):
                dist = np.linalg.norm(c - avg_px)
                if dist < best_dist:
                    best_dist = dist
                    best_ind = i
            groups[best_ind][f] = avg_px

        centers = [np.mean(list(groups[i].values()), axis=0) for i in range(k)]
    
    if save_pkl:
        with open('k_means_result.pkl', 'wb') as f:
            pickle.dump((centers, groups), f)
    return centers, groups

if __name__ == '__main__':
    import time
    time1 = time.time()
    if os.path.exists('avg_px.pkl'):
        with open('avg_px.pkl', 'rb') as f:
            images = pickle.load(f)
    else:
        images = import_from_folder("images", save_pkl=True)
    print(time.time() - time1)

    target_img = Image.open('421.png')
    target_img_np = np.asarray(target_img)

    ele_width = 32
    a, b, c = target_img_np.shape
    new_img_np = np.zeros((a*ele_width, b*ele_width, c))
    print(new_img_np.shape)
    if os.path.exists('k_means_result.pkl'):
        with open('k_means_result.pkl', 'rb') as f:
            centers, groups = pickle.load(f)
    else:
        centers, groups = k_means(images, k=20, num_iters=10, save_pkl=True)
    for i in tqdm(range(a)):
        for j in range(b):
            # alt_img_f = select_img(target_img_np[i, j], images, remove_used=True)
            alt_img_f = select_img_km(target_img_np[i, j], images, centers, groups, remove_used=False, rand=True)
            alg_img = Image.open('images/'+alt_img_f)
            alg_img_np = np.asarray(alg_img)
            new_img_np[i*ele_width:(i+1)*ele_width, j*ele_width:(j+1)*ele_width] = alg_img_np
    Image.fromarray(np.uint8(new_img_np)).save('result_random.png')
