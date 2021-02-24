import argparse
import os, sys
import pickle as pkl
from pycocotools.coco import COCO

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
from helpers import bbox

# For each ground truth, FN = all dets failed to cover more than 50% of total GT area
# For each detection, FP = failed to match a GT with more than 50% of its area
det_TP_thres = 0.3
gt_TP_thres = 0.3

frame_step = 20
video_nframe = 200

result_path = "./results/20210224-HOG-MSCM-LiFe"

def parse_args():
    parser = argparse.ArgumentParser(description='Extract positive and negative patches from a COCO-format dataset')
    parser.add_argument('--dataset-dir', help='The root path of a COCO-format dataset') 
    args = parser.parse_args()
    return args

def scoreAreaIntersection(det, gt):
    union = det.union(gt)
    if union is None:
        return 0
    else:
        return union.area() / det.area()
    

if __name__ == "__main__":

    args = parse_args()

    dataset = os.path.join(args.dataset_dir, "test")
    test_imgs = os.listdir(dataset)
    test_imgs.sort(key= lambda x:int(x[:-4]))

    coco = COCO(os.path.join(args.dataset_dir, 'annotations', 'VIS_Onshore_test.json'))

    tot_dets = 0
    tot_gts = 0

    horizon_cache = {}

    # Cache horizon detection for the same video
    index = 0
    for img_fname in test_imgs:
        index = index + 1
        i = (index - 1) // video_nframe
        if index % frame_step != 1:
            continue
        if index % video_nframe == 1:
            horizon_cache[i] = None

        filename, _ = os.path.splitext(img_fname)
        with open(os.path.join(result_path, filename + "_mscm_life_horizon.pkl"), "rb") as f:
            horizon = pkl.load(f)
            if horizon is not None:
                if abs(horizon.k) <= 1e2:
                    if horizon_cache[i] is None:
                        horizon_cache[i] = horizon
                    elif abs(horizon_cache[i].k) > abs(horizon.k):
                        horizon_cache[i] = horizon

    # Main evaluation loop

    FN = 0
    FP = 0

    index = 0
    for img_fname in test_imgs:

        index = index + 1
        if index % frame_step != 1:
            continue
        if index >= 1701:
            break

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
            det = bbox((det[0], det[1], det[2], det[3]))
            if horizon is not None:
                if not horizon.checkSuppress(det):
                    temp_dets.append(det)
            else:
                if dets[2] - dets[0] != 100 or dets[3] - dets[1] != 40:
                    temp_dets.append(det)
        dets = temp_dets
        
        ann_ids = coco.getAnnIds(imgIds=[int(filename)])
        ann_list = coco.loadAnns(ann_ids)
        gts = [bbox(tuple(ann["bbox"])) for ann in ann_list]
        tot_dets = tot_dets + len(dets)
        tot_gts = tot_gts + len(gts)

        for det in dets:
            TP = False
            for gt in gts:
                if scoreAreaIntersection(det, gt) >= det_TP_thres:
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

    


