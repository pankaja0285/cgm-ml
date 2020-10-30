import sys
# sys.path.append('.')
import dbutils
import pandas as pd
import yaml
from pyntcloud import PyntCloud
import utils
import numpy as np


def extract_qrcode(row):
    """ get the qrcode from the artifacts 

    Args:
        row (dataframe rows): complete row of a dataframe

    Returns:
        string: qrcodes extracted from the artifacts  
    """
    qrc = row['storage_path']
    split_qrc = qrc.split('/')[1]
    return split_qrc

class CollectQrcodes:
    """ class to gather a qrcodes from backend to prepare dataset for model training and evaluation.
    """

    def __init__(self,db_connector):
        """

        Args:
            db_connector (json_file): json file to connect to the database.
        """
        self.db_connector = db_connector
        self.ml_connector = dbutils.connect_to_main_database(self.db_connector)
    def get_all_data(self,realtime):
        """ Gather the qrcodes from the databse with labels.

        Args:
            realtime (bool): boolean value weather to prepare clean dataset or not.

        Returns:
            dataframe: panda dataframe contains data from databse.    
        """
        table_name = 'artifacts_with_target'
        columns = self.ml_connector.get_columns(table_name)
        query = "select * from " + table_name +';' 
        database = self.ml_connector.execute(query, fetch_all=True)
        database = pd.DataFrame(database,columns= columns)
        database['qrcode'] = database.apply(extract_qrcode,axis=1)
        if realtime:
            complete_data = database 
        else:
            complete_data = database[database['tag'] =='good']     
        return complete_data

    def get_training_data(self,data,full_dataset=True):
        """ get the training  data from database.
        Args:
            data (dataframe): panda dataframe containing the qrcode and labels.
            full_dataset(Boolean): boolean value to specify if all the training data to processed or just the remaining one.
        """

        if full_dataset:
            training_data = data[(data['scan_group']=='train')|(data['scan_group'].isnull())]
        else:
            training_data = data[data['scan_group'].isnull()]

        return training_data

    def get_evalaution_data(self,data,full_dataset=True):

        """ get the evaluation data from database.

        Args:
            data (dataframe): panda dataframe containing the qrcode and labels.
            full_dataset(Boolean): boolean value to specify if all the evaluation data to processed or just the remaining one
        """
        if full_dataset:
            evaluation_data = data[(data['scan_group']=='test')|(data['scan_group'].isnull())]
        else:
            evaluation_data = data[data['scan_group'].isnull()]

        return evaluation_data

    def get_unique_qrcode(self,dataframe):
        """
        Get a unique qrcodes from the dataframe file and return dataframe with qrcodes, scan_group, tags

        Args:
            dataframe (panda dataframe): A panda dataframe file with qrcodes, artifacts, and all the metadata.

        """
        data = dataframe.drop_duplicates(subset = ["qrcode"],keep='first') 
        unique_qrcode_data = data[['qrcode','scan_group','tag']]

        return unique_qrcode_data

    def get_usable_data(self,dataframe,amount,scan_group='train'):
        """[summary]

        Args:
            dataframe ([type]): [description]
            amount ([type]): [description]
        """
        available_data = dataframe[dataframe['scan_group'].isnull()]
        used_qrcodes =  dataframe[dataframe['scan_group'] == scan_group]
        bad_amount  = int(.4*amount)
        good_amount  = int(amount - bad_amount)
        used_good_qrcodes = used_qrcodes[used_qrcodes['tag'] == 'good']
        used_bad_qrcodes = used_qrcodes[used_qrcodes['tag'] == 'bad']
        required_good_amount = good_amount -len(used_good_qrcodes)
        if required_good_amount <= 0:
            print("Not enough good qrcodes present in database")
            return
        required_bad_amount = bad_amount -len(used_bad_qrcodes)

        if required_bad_amount <=0:
            print("Not enough bad qrcodes present in database")
            return
        new_good_qrcodes = available_data[available_data['tag'] =='good'].sample(n=required_good_amount)
        new_bad_qrcodes = available_data[available_data['tag'] =='bad'].sample(n=required_bad_amount)
        dataList = [used_qrcodes,new_good_qrcodes,new_bad_qrcodes]
        complete_data = pd.concat(dataList)
        return complete_data

    def merge_qrcode_dataset(self,qrcodes,dataset):
        """

        Args:
            qrcodes ([type]): [description]
            dataset ([type]): [description]
        """
        qrcodes = qrcodes['qrcode']
        full_dataset = pd.merge(qrcodes,dataset,on='qrcode',how='left')
        return full_dataset

    def get_posenet_results(self):
        """
        Fetch the posenet data for RGB and collect their ids.

        """
        artifact_result = "select * from artifact_result where artifact_id like '%_version_5.0%' and model_id ='posenet_1.0';" 
        artifacts_columns = self.ml_connector.get_columns('artifact_result')
        artifacts_table = self.ml_connector.execute(artifact_result, fetch_all=True)
        artifacts_frame = pd.DataFrame(artifacts_table,columns= artifacts_columns)
        artifacts_frame = artifacts_frame.rename(columns={"artifact_id": "id"})
        return artifacts_frame

    def  get_artifacts(self):
        """
        Get the artifacts results from the database for RGB
        """

        query = "select id,storage_path,qr_code from artifact where id like '%_version_5.0%' and dataformat ='rgb';" 
        artifacts= self.ml_connector.execute(query, fetch_all=True)
        artifacts = pd.DataFrame(artifacts,columns= ['id','storage_path','qrcode'])
        return artifacts

    def merge_data_artifacts(self,data,artifacts):
        """ Merge the two dataset of artifacts and posenet database 

        Args:
            data (dataframe):  dataframe with  qrcodes and all the other labels.
            artifacts (dataframe): artifacts.
        """

        rgb_data = pd.merge(data[['qrcode']],artifacts,on='qrcode',how='left')
        results = rgb_data.drop_duplicates(subset='storage_path', keep='first', inplace=False)
        results = results.drop_duplicates(subset='id', keep='first', inplace=False)
        return results
        
    def merge_data_posenet(self,data,posenet):
        """
        Merge the dataset and posenet datafarme to gather posenent results for available RGB

        Args:
            data (datafarme): dataframe with artifacts data
            posenet (dataframe): posenet data with keypoints for different bodypart.
        """
        posenet_results = pd.merge(data,posenet[['id','json_value','confidence_value']],on='id',how='left')
        return posenet_results
    
def load_pcd_as_ndarray(pcd_path):
    """[summary]

    Args:
        pcd_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    pointcloud = PyntCloud.from_file(pcd_path)
    values = pointcloud.points.values
    return values


## convert pointcloud to depthmaps
def lenovo_pcd2depth(pcd,calibration):
    """[summary]

    Args:
        pcd ([type]): [description]
        calibration ([type]): [description]

    Returns:
        [type]: [description]
    """
    points = utils.parsePCD(pcd)
    width = utils.getWidth()
    height = utils.getHeight()
    output = np.zeros((width, height, 1))
    for p in points:
        v = utils.convert3Dto2D(calibration[1], p[0], p[1], p[2])
        x = round(width - v[0] - 1)
        y = round(v[1])
        y = round(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x][y] = p[2]        
    return output 
