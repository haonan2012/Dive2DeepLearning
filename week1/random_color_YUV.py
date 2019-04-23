import cv2
import random

def random_color_yuv(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y,u,v = cv2.split(img_yuv)

    for c in [y,u,v]:
        rand_shift = random.randint(-60, 60)
        c[c > 255 - rand_shift] = 255 - rand_shift
        c[c < -rand_shift] = -rand_shift
        c[:] = c[:] + rand_shift

    img_merge = cv2.merge((y,u,v))
    img_bgr = cv2.cvtColor(img_merge, cv2.COLOR_YUV2BGR)
    return img_bgr

img = cv2.imread('1.jpg')
img_random_color = random_color_yuv(img)
cv2.imshow('img_ori', img)
cv2.imshow('img_random_color', img_random_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
