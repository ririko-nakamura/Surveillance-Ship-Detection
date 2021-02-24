class bbox:

    def __init__(self, det=()):
        if len(det) == 4:
            self.min_x = det[0]
            self.max_x = det[0] + det[2]
            self.min_y = det[1]
            self.max_y = det[1] + det[3]
        elif len(det) == 5:
            self.min_x = det[0]
            self.max_x = det[0] + det[3]
            self.min_y = det[1]
            self.max_y = det[1] + det[4]
        else:
            self.max_x = -1
            self.min_x = 65535
            self.max_y = -1
            self.min_y = 65535

    def area(self):
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def union(self, b):
        min_x = max(self.min_x, b.min_x)
        max_x = min(self.max_x, b.max_x)
        min_y = max(self.min_y, b.min_y)
        max_y = min(self.max_y, b.max_y)
        w = max_x - min_x
        h = max_y - min_y
        if w <= 0 or h <= 0:
            return None
        else:
            return bbox((min_x, min_y, w, h))

    def to_xywh(self):
        return (self.min_x, self.min_y, self.max_x - self.min_x, self.max_y - self.min_y)

    def __str__(self):
        return "[{}, {}, {}, {}]".format(self.min_x, self.min_y, self.max_x, self.max_y)

    __repr__ = __str__