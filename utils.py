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
    
def open_im_list(txt_im_list):
    if ".txt" not in txt_im_list:
        project = txt_im_list
        CONFIG = get_config(project)
        txt_im_list = CONFIG["im_list"]

    im_list = [line.rstrip() for line in open(txt_im_list, 'r')]
    return im_list