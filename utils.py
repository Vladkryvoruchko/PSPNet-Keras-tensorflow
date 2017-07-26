import colorsys
import numpy as np

def add_color(img):
    h,w = img.shape
    img_color = np.zeros((h,w,3))
    for i in xrange(1,151):
        img_color[img == i] = to_color(i)
    return img_color

def to_color(category):
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v,1,1)