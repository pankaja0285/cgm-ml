import sys
sys.path.append('.')
import dbutils
import pandas as pd
import numpy as np
from glob2 import glob
from azureml.core import Workspace, Dataset
import yaml
import random
import logging
import pickle
import os
import multiprocessing
import pathlib 
from data_utils import CollectQrcodes,lenovo_pcd2depth
import shutil
import utils

## Load the yaml file
with open("parameters.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)


# Parse all the configuration variables

db_file = cfg["database"]['db_connection_file']
training_file = cfg['csv_paths']['training_paths']
testing_file = cfg['csv_paths']['testing_paths']
number_of_scans =cfg['scans']['scan_amount']
calibration_file = cfg['calibration']['calibration_file']
dataset_name = cfg['data']['dataset']
target_folder = cfg['paths']['target_path']
realtime = cfg['scans']['realtime']
source = cfg['paths']['source_path']

pcd_path = target_folder+'pointclouds'
if not os.path.exists(pcd_path):
    os.makedirs(pcd_path)
    
depthmap_path = target_folder+'depthmaps'
if not os.path.exists(depthmap_path):
    os.makedirs(depthmap_path)

rgb_path = target_folder+'rgb'
if not os.path.exists(rgb_path):
    os.makedirs(rgb_path)

dataset = CollectQrcodes(db_file)
print("Starting the dataset Preparation:")
data = dataset.get_all_data(realtime=realtime)
evaluation_data = dataset.get_evalaution_data(data=data,full_dataset=True)
evaluation_qrcodes = dataset.get_unique_qrcode(evaluation_data)
new_evaluation_data = dataset.get_usable_data(dataframe=evaluation_qrcodes,amount=number_of_scans,scan_group='test')
print(new_evaluation_data)
# new_evaluation_data = new_evaluation_data[new_evaluation_data['tag'] != 'delete'] #TODO : check and then remove
full_dataset = dataset.merge_qrcode_dataset(new_evaluation_data,evaluation_data)

print("Saving the csv file for EDA notebook.")
full_dataset.to_csv('evaluation.csv',index=False)

## Create the RGB csv file for posenet.

get_posenet_data = dataset.get_posenet_results()
get_rgb_artifacts = dataset.get_artifacts()

get_rgb_qrcodedata = dataset.merge_data_artifacts(full_dataset,get_rgb_artifacts)
get_posenet_results = dataset.merge_data_posenet(full_dataset,get_posenet_data)

get_posenet_results.to_csv("RGB_poseresults.csv",index=False)

#Read the Calibration file and set the required shape fro height and width
calibration = utils.parseCalibration(calibration_file)
Width = utils.setWidth(int(240 * 0.75))
Height = utils.setHeight(int(180 * 0.75))


print("Processing the data")

def process_data(rows):
    source_path = source + rows['storage_path']
    qrcode = rows['qrcode']
    # qrcode = rows['storage_path'].split('/')[1]
    pcdfile = rows['storage_path'].split('/')[-1]
    depthmaps = lenovo_pcd2depth(source_path,calibration)
    max_value = depthmaps.max()
    if max_value > 10:
        logging.warning(pcdfile)
        return
    scantype = pcdfile.split('_')[3]
    pickle_file = pcdfile.replace('.pcd','.p')
    labels = np.array([float(rows['height']),float(rows['weight']),float(rows['muac']),rows['age'],rows['sex'],rows['tag'],rows['scan_group']])
    depthmap_target_path  = os.path.join(depthmap_path,qrcode)
    depthmap_complete_path = os.path.join(depthmap_target_path,scantype)
    pathlib.Path(depthmap_complete_path).mkdir(parents=True, exist_ok=True)
    data = (depthmaps,labels)
    depthmap_save_path = depthmap_complete_path+'/'+ pickle_file
    pickle.dump(data, open(depthmap_save_path, "wb"))
    
    pcd_target_path  = os.path.join(pcd_path,qrcode)
    pcd_complete_path = os.path.join(pcd_target_path,scantype)
    pathlib.Path(pcd_complete_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path,pcd_complete_path)
    return


def process_RGB(rows):
    source_path = source + rows['storage_path']
    qrcode = rows['qrcode']
    # qrcode = rows['storage_path'].split('/')[1]
    imagefile = rows['storage_path'].split('/')[-1]
    scantype = imagefile.split('_')[3]    
    rgb_target_path  = os.path.join(rgb_path,qrcode)
    rgb_complete_path = os.path.join(rgb_target_path,scantype)
    pathlib.Path(rgb_complete_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path,rgb_complete_path)
    return
    
proc = multiprocessing.Pool()

for index, row in full_dataset.iterrows():
    process_data(row)
    # proc.apply_async(process_pcd, [row]) 
    

# for files in datas:
#     # launch a process for each file (ish).
#     # The result will be approximately one process per CPU core available.
#     proc.apply_async(process_file, [files]) 

proc.close()
proc.join() # Wait for all child processes to close.

for index, row in full_dataset.iterrows():
    process_RGB(row)
    # proc.apply_async(process_RGB, [row]) 
    

# for files in datas:
#     # launch a process for each file (ish).
#     # The result will be approximately one process per CPU core available.
#     proc.apply_async(process_file, [files]) 

proc.close()
proc.join()
    
# mount_context.stop() ## stop the mounting stream

