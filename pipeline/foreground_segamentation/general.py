import cv2 as cv

class detection:

    def __init__(self):
        self.max_x = -1
        self.min_x = 65535
        self.max_y = -1
        self.min_y = 65535

    def __str__(self):
        return "[{}, {}, {}, {}]".format(self.min_x, self.min_y, self.max_x, self.max_y)

    __repr__ = __str__

def extractDetections(mask):

    max_mask_id = 0
    detections = []

    # Wrapping mask
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            cur_id = mask[y, x]
            if cur_id == 0:
                continue
            if cur_id > max_mask_id:
                while max_mask_id != cur_id:
                    detections.append(detection())
                    max_mask_id = max_mask_id + 1
                assert max_mask_id == cur_id, "max_mask_id != cur_id, ids are not continious or begins from 1"
            index = cur_id - 1
            detections[index].max_x = max(detections[index].max_x, x)
            detections[index].min_x = min(detections[index].min_x, x)
            detections[index].max_y = max(detections[index].max_y, y)
            detections[index].min_y = min(detections[index].min_y, y)

    return detections

class ForegroundSegamentor:

    def __init__(self, area_th = 512):
        self.area_th = area_th
        self.kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        self.kernel3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.kernel5 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    def apply(self, mask):

        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel2, iterations=1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel3, iterations=3)

        _, labels, stats, _ = cv.connectedComponentsWithStats(mask)

        cur_id = 1
        id_pairs = []
        for i in range(stats.shape[0]):
            area = stats[i, cv.CC_STAT_AREA]
            if i != 0 and area >= self.area_th:
                id_pairs.append((cur_id, i))
                cur_id = cur_id + 1

        for x in range(labels.shape[1]):
            for y in range(labels.shape[0]):
                found = False
                for id_pair in id_pairs:
                    if labels[y, x] == id_pair[1]:
                        labels[y, x] = id_pair[0]
                        found = True
                if not found:
                    labels[y, x] = 0

        return extractDetections(labels)
