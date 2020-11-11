# Create dataset utility

Dataset creation utility is used to create datasets for model training and evaluation.

## Overview

CGM-ML infrastructre currently prepares the dataset into 3 formats(RGB,pointclouds,depthmaps). PCD file is a standardised format. Depthmap is our own format developed for high compressed data.

In the future we plan to not support PCD files anymore (due to their big size).

## Configuration

* The utility  looks for the configuration parameters provided in the `parameters.yaml`.
- `db_connection_file`: connection file to connect to the database 
- `scan_group`: type of dataset you want to prepare ['train','test']
- `scan_amount` : Amount of the data that you want to process.
- `source_path` : Source path from where the utility will get the data to process
- `target_path`: Target paths to storte the prepared data.

### Converting depthmaps into PCD data

* The convertor accepts only the data captured by cgm-scanner. The data could be captured by any ARCore device supporting ToF sensor. Converting could be done by following command:

python convertdepth2pcd.py input

* The input folder has to contain camera_calibration.txt file and subfolder depth containing one or more depthmap files.
* The output will be stored in folder export.


