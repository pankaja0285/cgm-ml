import os
import pickle

import glob2 as glob
import numpy as np
import pandas as pd

from config import DATA_CONFIG, RESULT_CONFIG

def preprocess_targets(targets, targets_indices):
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")

def getFilename(imPath):
    fName = ''
    fNameArr = imPath.split("/scans/")
    fName = fNameArr[1]
    return fName

def get_rgb_files(paths):
    '''
    Prepare the list of all the rgb files in dataset
    '''
    rgb_paths = []
    for path in paths:
        rgb_paths.extend(glob.glob(os.path.join(path, "**", "*.jpg")))
    return rgb_paths


def get_column_list(rgb_path_list):
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

def avgerror(row):
    difference = row['GT'] - row['predicted']
    return difference


def calculate_performance(code, df_mae):
    '''
    For each scantype, calculate the performance of the model
    across all error margin
    '''
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    for acc in RESULT_CONFIG.ACCURACIES:
        good_predictions = df_mae_filtered[(df_mae_filtered['error'] <= acc) & (df_mae_filtered['error'] >= -acc)]
        if len(df_mae_filtered):
            accuracy = len(good_predictions) / len(df_mae_filtered) * 100
        else:
            accuracy = 0.
        # print(f"Accuracy {acc:.1f} for {code}: {accuracy}")
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = RESULT_CONFIG.ACCURACIES
    return df_out


def calculate_and_save_results(MAE, complete_name, CSV_OUT_PATH):
    '''
    Calculate accuracies across the scantypes and
    save the final results table to the CSV file
    '''
    dfs = []
    for code in DATA_CONFIG.CODE_TO_SCANTYPE.keys():
        df = calculate_performance(code, MAE)
        full_model_name = complete_name + DATA_CONFIG.CODE_TO_SCANTYPE[code]
        df.rename(index={0: full_model_name}, inplace=True)
        #display(HTML(df.to_html()))
        dfs.append(df)

    result = pd.concat(dfs)
    result.index.name = 'Model_Scantype'
    result = result.round(2)
    print(result)

    # Save the model results in csv file
    result.to_csv(CSV_OUT_PATH, index=True)

    
