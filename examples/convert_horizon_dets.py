# This script will convert results from Surveillance-Ship-Detection
# results in pkl format to a json.

import json
import os, sys
import argparse
import pickle as pkl
import bisect

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../")
from pipeline.general import Horizon

parser = argparse.ArgumentParser()
parser.add_argument("results_path", help="File path to Surveillance-Ship-Detection result path")
parser.add_argument("detector", help="Horizon detector used when producing the results")
parser.add_argument("dataset_size", type=int, help="Total count of images in the target dataset")
parser.add_argument("seg_size", type=int, help="""Count of images within a video segment. Interpolation
                                        will not be conducted between segments""")
parser.add_argument("--img_w", type=int, help="Image height in dataset", default=1920)
parser.add_argument("--img_h", type=int, help="Image width in dataset", default=1080)

def check_valid(det, img_shape):
    A = (0, det.y(0))
    B = (img_shape[1] - 1, det.y(img_shape[1] - 1))
    if abs(A[1]) > 1e4 or abs(B[1]) > 1e4:
        return False
    else:
        return True

def NEED_SKIP(dataset_name, i):
    # Define your skip list and remove the following statement
    return False
    if "train" in dataset_name:
        if i > 2100 and i <= 2400:
            return True
        if i > 2700 and i <= 3000:
            return True
        if i > 3600 and i <= 3900:
            return True
        return False
    elif "test" in dataset_name:
        if i > 1200 and i <= 1800:
            return True
        return False
    assert False, "Database type not defined (train / val / test)"

if __name__ == "__main__":

    args = parser.parse_args()
    dataset_name = os.path.basename(args.results_path)

    indexes_with_det = []
    det_dict = {}
    detector_str = args.detector.replace('-','_').lower()
    for i in range(1, args.dataset_size + 1):
        det_path = os.path.join(args.results_path, str(i) + "_" + detector_str + "_horizon.pkl")
        if os.path.exists(det_path):
            with open(det_path, "rb") as f:
                horizon = pkl.load(f)
                if horizon is None or not check_valid(horizon, (args.img_w, args.img_h)):
                    continue
                else:
                    indexes_with_det.append(i)
                    det_dict[i] = horizon

    for i in range(1, args.dataset_size + 1):

        if NEED_SKIP(dataset_name, i):
            continue

        seg_i = (i - 1) // args.seg_size
        l = args.seg_size * seg_i + 1
        r = args.seg_size * (seg_i + 1) + 1
        p = bisect.bisect_right(indexes_with_det, i)
        if p == 0 or indexes_with_det[p - 1] < l:
            l = None
        else:
            l = indexes_with_det[p - 1]
        if p == len(indexes_with_det) or indexes_with_det[p] >= r:
            r = None
        else:
            r = indexes_with_det[p]

        assert l is not None or r is not None, "Image with index {} doesn't have a reference horizon det".format(i)
        if l is None:
            det_dict[i] = det_dict[r]
        elif r is None:
            det_dict[i] = det_dict[l]
        else:
            det_dict[i] = Horizon.interpolate(i, l, det_dict[l], r, det_dict[r])

        print("=====================")
        print(i, det_dict[i].point, det_dict[i].k)
        if l is None:
            print("None")
        else:
            print(l, det_dict[l].point, det_dict[l].k)
        if r is None:
            print("None")
        else:
            print(r, det_dict[r].point, det_dict[r].k)

    json_obj = {}
    json_obj["detector"] = args.detector
    json_obj["detections"] = []
    for i in range(1, args.dataset_size + 1):
        if NEED_SKIP(dataset_name, i):
            continue
        json_obj["detections"].append({
            "image_id": i,
            "horzion": det_dict[i].toArray()
        })
    with open("{}_{}_horizon_dets.json".format(dataset_name, args.detector), "w") as f:
        json.dump(json_obj, f)
    