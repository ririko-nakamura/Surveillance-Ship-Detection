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

    def __str__(self):
        return "[{}, {}, {}, {}]".format(self.min_x, self.min_y, self.max_x, self.max_y)

    __repr__ = __str__