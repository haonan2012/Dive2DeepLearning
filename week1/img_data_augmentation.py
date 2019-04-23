import cv2
import random
import numpy as np
import os

def random_light_color(img, shift):
    # brightness
    if len(img.shape) == 2:
        return img
    B, G, R = cv2.split(img)
    for C in [B,G,R]:
        rand_shift = random.randint(-shift, shift)
        C[C > 255 - rand_shift] = 255 - rand_shift
        C[C < -rand_shift] = -rand_shift
        C[:] = C[:] + rand_shift
    return cv2.merge((B, G, R))

def random_perspective_transform(img):
    height, width = img.shape[:2]
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp

def img_data_augmentation(img, crop_ratio = 1.0, color_shift = 0, rotation = 0, perspective_transform = False):
    assert crop_ratio > 0 and crop_ratio <= 1.0
    assert len(img.shape) == 2 or len(img.shape) == 3
    height, width = img.shape[:2]

    if crop_ratio < 1.0:
        crop_height, crop_width = int(height * crop_ratio), int(width * crop_ratio)
        y = random.randint(0, height - crop_height)
        x = random.randint(0, width - crop_width)
        if len(img.shape) == 2:
            img = img[y:y+crop_height, x:x+crop_width]
        if len(img.shape) == 3:
            img = img[y:y+crop_height, x:x+crop_width, :]
    
    if color_shift != 0:
        img = random_light_color(img, color_shift)
    
    if rotation != 0:
        rotation_angle = random.randint(-rotation, rotation)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotation_angle, 1.0) 
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    if perspective_transform == True:
        img = random_perspective_transform(img)

    return img

img = cv2.imread('1.jpg', 0)

save_path = 'gene_data'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(20):
    img_random = img_data_augmentation(img, crop_ratio = 0.8, color_shift=30, rotation=40, perspective_transform=True)
    img_name = save_path + '\\random_img_' + str(i) + '.jpg'
    cv2.imwrite(img_name, img_random)

