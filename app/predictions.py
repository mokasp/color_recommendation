#!/usr/bin/env python3
from .colorspace import process_color_lists
from .data_processing import get_all_regions, unnormalize_vector, process_regions
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import numpy as np
import logging

def predict(img, face_mesh, model):
    logging.debug('hit0')
    input_regions, output_regions, region_names = get_all_regions(img, face_mesh)
    logging.debug('hit1')
    norm_input_vectors_lab, input_vectors_lab, norm_output_lab, output_lab = process_regions(input_regions, output_regions, region_names)
    logging.debug('hit2')
    norm_input_vectors_lab = np.expand_dims(norm_input_vectors_lab, axis=0)
    logging.debug('hit3')
    logging.debug(norm_input_vectors_lab)
    prediction = model.predict(norm_input_vectors_lab[0])
    logging.debug('hit4')

    return prediction, input_vectors_lab, output_lab

def find_best_color(unnorm_vector, color_lists_lab, color_set=9):
    pred_vector = LabColor(float(unnorm_vector[0]), float(unnorm_vector[1]), float(unnorm_vector[2]))
    distances = []
    if color_set == 9:
        for vectors in color_lists_lab:
            for vector in vectors:
                distance = float(delta_e_cie2000(pred_vector, vector))
                distances.append(distance)
        min_distance = min(distances)
        min_index = distances.index(min_distance)

        i = int(min_index /  3)
        j = int(min_index % 3)

    else:
        i = color_set
        for vector in color_lists_lab[i]:
            distance = float(delta_e_cie2000(pred_vector, vector))
            distances.append(distance)
        min_distance = min(distances)
        min_index = distances.index(min_distance)

        j = int(min_index % 3)
    
    return i, j

def get_best_color(prediction, color_lists, color_set=9):
    color_lists_lab = process_color_lists(color_lists)
    unnorm_vector = unnormalize_vector(prediction)
    i, j = find_best_color(unnorm_vector, color_lists_lab, color_set=color_set)

    best_color = color_lists[i][j]
    return best_color