#!/usr/bin/env python3
from .colorspace import get_lab_vector, combine_lab_values
import cv2
import numpy as np
import logging

def direction(img, h, w, face_results):
    """ determines direction person is facing in the photo to extract the correct region for hair """
    # get the facial landmarks
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0]

        # extract the necessary landmarks (nose tip, left eye, right eye)
        nose_tip = landmarks.landmark[2]
        left_eye = landmarks.landmark[133]
        right_eye = landmarks.landmark[33]

        # convert to pixel coordinates
        fx = int(nose_tip.x * w)
        fy = int(nose_tip.y * h)
        le_x = int(left_eye.x * w)
        le_y = int(left_eye.y * h)
        re_x = int(right_eye.x * w)
        re_y = int(right_eye.y * h)

        # calculate the horizontal distance between the nose tip and each eye
        nose_to_left_eye = abs(fx - le_x)
        nose_to_right_eye = abs(fx - re_x)

        # calculate vertical distances (y-coordinates)
        nose_to_left_eye_y = abs(fy - le_y)
        nose_to_right_eye_y = abs(fy - re_y)

        # set thresholds for detecting left, right, and forward
        horizontal_threshold = 30
        vertical_threshold = 40   # threshold for y-coordinates to distinguish tilt

        # determine face orientation based on distances
        if nose_to_left_eye > horizontal_threshold and nose_to_left_eye > nose_to_right_eye and nose_to_left_eye_y < vertical_threshold:
            orientation = "LEFT"
        elif nose_to_right_eye > horizontal_threshold and nose_to_right_eye > nose_to_left_eye and nose_to_right_eye_y < vertical_threshold:
            orientation = "RIGHT"
        else:
            # set thresholds for detecting left, right, and forward
            horizontal_threshold = 30
            vertical_threshold = 30  # Threshold for y-coordinates to distinguish tilt
            # checking if the face is mostly forward (both horizontal and vertical distances are small)
            if nose_to_left_eye < horizontal_threshold and nose_to_right_eye < horizontal_threshold and \
              nose_to_left_eye_y < vertical_threshold and nose_to_right_eye_y < vertical_threshold:
                orientation = "FORWARD"
            else:
                # if face is slightly turned but not clear, assume forward-facing
                orientation = "LEFT"

        return orientation, face_results

def extract_hair(img, dir, h, w, face_results):
    """ extract region from top of head and side of head to determine hair color """
    # define list for storing input regions and their names (for visualization)
    input_regions = []
    region_names = ["00_side_hair", "01_top_hair"]

    if face_results.multi_face_landmarks:
        # get all landmarks and landmark for the forehead
        landmarks = face_results.multi_face_landmarks[0]
        forehead = landmarks.landmark[10]
        fx = int(forehead.x * w)
        fy = int(forehead.y * h)

        # --- Top Hair Box ---
        top = max(fy - 35, 0)
        bottom = max(fy - 20, 0)
        left = int(w * 0.3)
        right = int(w * 0.6)

        # --- Side Hair Box ---
        if dir == 'RIGHT':
            top_s = max(fy - 15, 0)
            bottom_s = max(fy - 30, 0)
            left_s = int(w * 0.2)
            right_s = int(w * 0.4)
        else:
            top_s = max(fy - 15, 0)
            bottom_s = max(fy - 30, 0)
            left_s = int(w * 0.5)
            right_s = int(w * 0.7)

        # extract the top and side hair regions from the image
        top_hair = img[top:bottom, left:right]
        side_hair = img[top:bottom, left_s:right_s]

        input_regions.append(top_hair)
        input_regions.append(side_hair)


    return input_regions, region_names

def extract_regions(image, h, w, results, input_regions, region_names):
    """  extracts regions from different facial features to determine undertones """
    output_regions = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            # Define Region Indices
            regions = {
                "02_left_eyelid": [225, 30, 29, 27],
                "03_right_eyelid": [445, 260, 259, 257],
                "04_nose_left_side": [189, 221, 193, 55],
                "05_nose_right_side": [413, 441, 417, 285],
                "06_nose_bridge": [197, 196, 195, 419],
                "07_nose_tip": [5, 281, 275, 1, 45, 51, 5],
                "08_forehead": [109, 10, 338, 299, 9, 69, 109],
                "09_left_cheek": [119, 100, 142, 203, 206, 207, 187, 117, 118, 119],
                "10_right_cheek": [348, 329, 371, 423, 426, 427, 411, 346, 347, 348],
                "11_chin": [175, 396, 428, 421, 313, 18, 83, 201, 208, 171, 175],
                "12_top_lips": [81, 82, 13, 312, 311, 267, 0, 37, 81],
                "13_bottom_lips": [181, 84, 17, 314, 405, 403, 317, 14, 87, 181]
            }


            # Create separate cropped images of each region
            for name, indices in regions.items():
                # Get the coordinates for the region
                pts = np.array([(int(face_landmarks.landmark[i].x * w),
                                int(face_landmarks.landmark[i].y * h)) for i in indices])

                # Compute bounding box
                x, y, w_box, h_box = cv2.boundingRect(pts)
                padding = 0
                x = max(0, x - padding)
                y = max(0, y - padding)
                w_box = min(image.shape[1] - x, w_box + 2 * padding)
                h_box = min(image.shape[0] - y, h_box + 2 * padding)

                # Crop the region
                cropped_region = image[y:y+h_box, x:x+w_box]
                cropped_region = cv2.resize(cropped_region, (0, 0), fx=10, fy=10)
                region_names.append(name)
                # Save the cropped region
                if name == "12_top_lips" or name == "13_bottom_lips":
                    output_regions.append(cropped_region)
                else:
                    input_regions.append(cropped_region)

    return input_regions, output_regions, region_names

def normalize_vector(lab_vector):

    true_L, true_a, true_b = lab_vector
    normalized_L = true_L / 100.0
    normalized_a = (true_a + 128) / 255.0
    normalized_b = (true_b + 128) / 255.0

    normalized_vector = [normalized_L, normalized_a, normalized_b]
    normalized_vector = np.array(normalized_vector)

    return normalized_vector

def unnormalize_vector(normalized_vector):
    """
    pred: array of shape (3,) or (1, 3)
    Returns: Lab vector in original range
    """
    if normalized_vector.ndim == 2:
        normalized_vector = normalized_vector[0] # remove batch dimension

    un_L = normalized_vector[0] * 100.0
    un_a = normalized_vector[1] * 255.0 - 128
    un_b = normalized_vector[2] * 255.0 - 128
    return [un_L, un_a, un_b]

def process_regions(input_regions, output_regions, region_names):

    input_vectors_lab = []
    norm_input_vectors_lab = []
    lip_lab_vectors = []

    i = 0
    for region in input_regions:
        lab_vector = get_lab_vector(region, region_names[i])
        norm_input_vector = normalize_vector(lab_vector)
        input_vectors_lab.append(lab_vector)
        norm_input_vectors_lab.append(norm_input_vector)
        i += 1

    for region in output_regions:
        lip_lab_vector = get_lab_vector(region, region_names[i])
        lip_lab_vectors.append(lip_lab_vector)
        i += 1
    logging.debug(lip_lab_vectors)
    output_lab = combine_lab_values(lip_lab_vectors[0], lip_lab_vectors[1])

    norm_input_vectors_lab = np.array(norm_input_vectors_lab)
    norm_output_lab = normalize_vector(output_lab)

    return norm_input_vectors_lab, input_vectors_lab, norm_output_lab, output_lab

def get_all_regions(img, face_mesh):
    """ extract all regions from image and store each region in a list """
    # convert image from bgr color space to rgb and get image dimensions
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape


    # initialize MediaPipe Face Mesh for facial regions
    face_mesh = face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # run face mesh
    face_results = face_mesh.process(rgb_img)
    face_dir = direction(img, h, w, face_results)
    input_regions, region_names = extract_hair(img, face_dir, h, w, face_results)
    input_regions, output_regions, region_names = extract_regions(img, h, w, face_results, input_regions, region_names)

    return input_regions, output_regions, region_names