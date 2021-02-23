import os, sys
import pickle as pkl
from pycocotools.coco import COCO

# For each ground truth, FN = all dets failed to cover more than 50% of total GT area
# For each detection, FP = failed to match a GT with more than 50% of its area
det_TP_thres = 0.5
gt_TP_thres = 0.5

frame_step = 20
video_nframe = 200

def loadGTs(coco, img_id):
    
    pass

def scoreAreaIntersection(gt, det):
    pass

if __name__ == "__main__":

    dataset = os.argv[1]
    result_path = "./results/20210223-HOG-MSCM-LiFe"
    test_imgs = os.listdir(dataset)
    test_imgs.sort(key= lambda x:int(x[:-4]))

    coco = COCO()

    tot_dets = 0
    tot_gts = 0

    horizon_cache = {}

    index = 0

    # Cache horizon detection for the same video
    for img_fname in test_imgs:
        index = index + 1
        i = (index - 1) // video_nframe
        if index % frame_step != 1:
            continue
        if index % video_nframe == 1:
            horizon_cache[i] = None
        with open(os.path.join(dataset, fname + "_mscm_life_horizon.pkl"), "rb") as f:
            horizon = pkl.load(f)
            if horizon is not None:
                horizon_cache[i] = horizon

    # Main evaluation loop
    for img_fname in test_imgs:

        index = index + 1
        if index % frame_step != 1:
            continue

        filename, _ = os.path.splitext(img_fname)
        with open(os.path.join(result_path, filename + "_mscm_life_detections.pkl"), "rb") as f:
            dets = pkl.load(f)
        with open(os.path.join(result_path, filename + "_mscm_life_horizon.pkl"), "rb") as f:
            horizon = pkl.load(f)
        if horizon is None:
            horizon = horizon_cache[(index - 1) // video_nframe]

        # Do horizon / non-maximal suppression
        temp_dets = []
        for det in dets:
            if horizon is not None:
                if not horizon.checkSuppress(det):
                    temp_dets.append(det)
            else:
                if dets[2] - dets[0] != 100 or dets[3] - dets[1] != 40:
                    temp_dets.append(det)
        dets = temp_dets
        
        gts = loadGTs()
        tot_dets = tot_dets + len(dets)
        tot_gts = tot_gts + len(gts)

        FN = 0
        FP = 0

        for det in dets:
            TP = False
            for gt in gts:
                if evaluateDetection(det, gt) >= det_TP_thres:
                    TP = True
                    break
            if not TP:
                FP = FP + 1

        for gt in gts:
            TP = False
            for det in dets:
                if scoreAreaIntersection(gt, det) >= gt_TP_thres:
                    TP = True
                    break
            if not TP:
                FN = FN + 1

    precision = 1 - FP / tot_dets
    recall = 1 - FN / tot_gts
    print("Precision = {0}, Recall = {1}".format(precision, recall))

    


