import os
import pickle

import glob2 as glob
import numpy as np
import pandas as pd

from config import DATA_CONFIG, RESULT_CONFIG

def _getFilename(imPath):
    fName = ''
    fNameArr = imPath.split("/scans/")
    fName = fNameArr[1]
    return fName

def _get_rgb_files(paths):
    '''
    Prepare the list of all the rgb files in dataset
    '''
    rgb_paths = []
    for path in paths:
        rgb_paths.extend(glob.glob(os.path.join(path, "**", "*.jpg")))
    return rgb_paths


def _get_column_list(rgb_path_list):
    '''
    Prepare the list of all artifact with its corresponding scantype,
    qrcode, target and prediction
    '''
    qrcode_list, artifact_list = [], []

    for idx, path in enumerate(rgb_path_list):
        sub_folder_list = path.split('/')
        qrcode_list.append(sub_folder_list[-3])
        artifact_list.append(sub_folder_list[-1])
        
    return qrcode_list, artifact_list
