import cv2 
import pandas as pd
import time
import os


def _init(proto, model):
    global net
    print('proto ', proto)
    
    net = cv2.dnn.readNetFromCaffe(proto, model)
    print('cv2 dnn readNetFromCaffe')
    return net

def _setPoseDetails(datasetType):
    body_parts = {}
    pose_pairs = []
    defaultDatasetType = 'default-dataset'

    if datasetType == 'COCO':
        body_parts = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        pose_pairs = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    elif datasetType=='MPI':
        body_parts = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                        "Background": 15 }

        pose_pairs = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                        ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                        ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                        ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    else:
        datasetType = defaultDatasetType
        body_parts ={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,"LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,"LKnee":13,"LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,"LSmallToe":20,"LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background":25}

        pose_pairs =[ ["Neck","MidHip"],   ["Neck","RShoulder"],   ["Neck","LShoulder"],   ["RShoulder","RElbow"],   ["RElbow","RWrist"],   ["LShoulder","LElbow"],   ["LElbow","LWrist"],   ["MidHip","RHip"],   ["RHip","RKnee"],  ["RKnee","RAnkle"], ["MidHip","LHip"],  ["LHip","LKnee"], ["LKnee","LAnkle"],  ["Neck","Nose"],   ["Nose","REye"], ["REye","REar"],  ["Nose","LEye"], ["LEye","LEar"],   
    ["RShoulder","REar"],  ["LShoulder","LEar"],   ["LAnkle","LBigToe"],["LBigToe","LSmallToe"],["LAnkle","LHeel"], ["RAnkle","RBigToe"],["RBigToe","RSmallToe"],["RAnkle","RHeel"] ]

    dataset_type_and_model = datasetType + '-caffemodel'
    return dataset_type_and_model, body_parts, pose_pairs

def _addColumnsToDataframe(body_parts, pose_pairs, df):
    pairCols = []
    
    for i in range(len(pose_pairs)):
        partFrom = pose_pairs[i][0]
        partTo = pose_pairs[i][1]
        assert(partFrom in body_parts)
        assert(partTo in body_parts)

        idFrom = body_parts[partFrom]
        idTo = body_parts[partTo]
        colname = "P"+str(idFrom)+str(idTo)
        pairCols.append(colname)
                
    # add other columns to the dataframe 
    df = pd.DataFrame(columns = df.columns.tolist() + pairCols)

    return df, pairCols    

def _poseEstimate(imagePath, net, body_parts, pose_pairs, threshold=0.1, width=368, height=368):
    points = []
    start_t = time.time()
    
    try:
        # print('imagePath - ', imagePath)
        imgRead = cv2.imread(imagePath)
        frame = cv2.rotate(imgRead, cv2.cv2.ROTATE_90_CLOCKWISE)
        pts = []
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height),
                                    (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        # do the forward-pass
        out = net.forward()

        print(f"time is {time.time()-start_t}")

        for i in range(len(body_parts)):
            # Slice heatmap of corresponding body part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            pts.append((int(x), int(y)) if conf > threshold else None)
        
        for pair in pose_pairs:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in body_parts)
            assert(partTo in body_parts)

            idFrom = body_parts[partFrom]
            idTo = body_parts[partTo]
            if pts[idFrom] and pts[idTo]:
                # so we keep the points (belonging to pairs together)
                currSetofPoints = [pts[idFrom], pts[idTo]]

                # NOTE: uncomment for DEBUG purposes
                # print('currSetofPoints for ', idFrom, idTo)
                # print('currSetofPoints ', currSetofPoints)

                points.append(currSetofPoints)
                # print('points ', points)
                                
                # the last pairs, are
                # - right ankle to right big toe 
                # - left ankle to left big toe
            else:
                # append [] if no pose points estimated
                points.append([])
                
    except Exception:
        filename = 'outputs/notprocessed.txt'

        if os.path.exists(filename):
            append_write = 'a' 
        else:
            append_write = 'w' 

        expwrite = open(filename, append_write)
        expwrite.write(f"File: {imagePath}\n")
        expwrite.close() 
    return points
