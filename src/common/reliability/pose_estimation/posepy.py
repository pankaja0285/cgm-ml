import cv2 
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

def _poseEstimationFromModelAndDataset(net, body_parts, pose_pairs, dataset_type_and_model,
 imagePath, imageFilename, getTitleboxAndResizeFrame,
 threshold=0.1, width=368, height=368, writeImage=False):

    imgRead = cv2.imread(imagePath)
    frame = cv2.rotate(imgRead, cv2.cv2.ROTATE_90_CLOCKWISE)
    #print('imageFilename: ', imageFilename)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height),
            (0, 0, 0), swapRB = False, crop = False)
    net.setInput(inp)
    start_t = time.time()
    out = net.forward()

    print("time is ",time.time() - start_t)
    #assert(len(body_parts) == out.shape[1])

    points = []
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
        points.append((int(x), int(y)) if conf > threshold else None)
        
    for pair in pose_pairs:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in body_parts)
        assert(partTo in body_parts)

        idFrom = body_parts[partFrom]
        idTo = body_parts[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
            cv2.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            
    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # set up titlebox to show some description etc. of the result
    # and also modify the outer edge dims of the frame
    frameWBorder, titlebox = getTitleboxAndResizeFrame(frame, 
                                imageFilename, additionalTitleText = dataset_type_and_model)
    # vconcat for combining the titlebox and frameWBorder
    imframe = cv2.vconcat((titlebox, frameWBorder))
    # set Pose Estimation file name for imwrite
    impath = f"PoseEst-{imageFilename}" 
    if (writeImage == True):
        #create folder if need be
        if (os.path.exists('output') == False):
            os.makedirs('output', mode = 0o777, exist_ok=False)
        # write the file
        cv2.imwrite(f"./output/{impath}", imframe)
        
    #display image
    plt.figure()
    img = cv2.cvtColor(imframe, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
    display(Image.fromarray(img))
    
    return points

