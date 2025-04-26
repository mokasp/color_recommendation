#!/usr/bin/env python3
import numpy as np
import cv2
import logging
import skimage
from colormath.color_objects import LabColor

def get_lab_vector(img, region_name):
    """ extracts lab vector from a single region """

    # convert region into Lab color space
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab_img = skimage.color.rgb2lab(rgb_img)

    # extract L, a, b channels
    L_channel = lab_img[:, :, 0]
    a_channel = lab_img[:, :, 1]
    b_channel = lab_img[:, :, 2]

    # calculate mean for each channel
    mean_L = np.mean(L_channel)
    mean_a = np.mean(a_channel)
    mean_b = np.mean(b_channel)

    lab_vector = [mean_L, mean_a, mean_b]

    return lab_vector

def combine_lab_values(top_lab, bottom_lab):
    """ combine the lab ectors of top and bottom lip to get the lip color """
    lab_combined = [
        (top_lab[0] + bottom_lab[0]) / 2,
        (top_lab[1] + bottom_lab[1]) / 2,
        (top_lab[2] + bottom_lab[2]) / 2]

    return lab_combined

def load_colors():
    colors = {}
    colors['bright_red'] = [[210, 50, 60], [220, 40, 30], [215, 45, 45]]
    colors['deep_red'] = [[130, 20, 40], [120, 25, 30], [125, 22, 35]]
    colors['light_red'] = [[245, 110, 120], [250, 90, 80], [248, 100, 100]]

    colors['bold_pink'] = [[230, 40, 100], [240, 60, 90], [235, 50, 95]]
    colors['soft_pink'] = [[255, 175, 200], [255, 155, 170], [255, 165, 185]]
    colors['nude_pink'] = [[220, 170, 180], [225, 155, 165], [223, 160, 172]]

    colors['nude'] = [[200, 170, 160], [205, 155, 140], [202, 160, 150]]
    colors['brown'] = [[120, 80, 70], [140, 90, 60], [130, 85, 65]]
    colors['coral'] = [[255, 120, 130], [255, 100, 80], [255, 110, 105]]


    color_lists = [
        [[210, 50, 60], [220, 40, 30], [215, 45, 45]],
        [[130, 20, 40], [120, 25, 30], [125, 22, 35]],
        [[245, 110, 120], [250, 90, 80], [248, 100, 100]],
        [[230, 40, 100], [240, 60, 90], [235, 50, 95]],
        [[255, 175, 200], [255, 155, 170], [255, 165, 185]],
        [[220, 170, 180], [225, 155, 165], [223, 160, 172]],
        [[200, 170, 160], [205, 155, 140], [202, 160, 150]],
        [[120, 80, 70], [140, 90, 60], [130, 85, 65]],
        [[255, 120, 130], [255, 100, 80], [255, 110, 105]]
    ]
    logging.debug(str(color_lists))
    return colors, color_lists

def rgb_to_lab(rgb_vector):
    rgb_vector = np.array(rgb_vector)
    rgb_vector = rgb_vector.reshape(1, 1, 3) / 255.0
    lab_vector = skimage.color.rgb2lab(rgb_vector)

    return lab_vector[0, 0]

def lab_to_rgb(lab_vector):
    lab_vector = np.array(lab_vector)
    lab_vector = lab_vector.reshape(1, 1, 3)
    rgb_vector = skimage.color.lab2rgb(lab_vector)

    return (rgb_vector[0, 0] * 255).astype(np.uint8)

def process_color_lists(color_lists):
    color_lists_lab = []
    for color_list in color_lists:
        lab_colors = []
        for vector in color_list:
            lab_vector = rgb_to_lab(vector)
            lab_vector = LabColor(float(lab_vector[0]), float(lab_vector[1]), float(lab_vector[2]))
            lab_colors.append(lab_vector)
        color_lists_lab.append(lab_colors)

    return color_lists_lab