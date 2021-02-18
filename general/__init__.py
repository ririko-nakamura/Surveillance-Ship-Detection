class bbox:

    def __init__(self):
        self.max_x = -1
        self.min_x = 65535
        self.max_y = -1
        self.min_y = 65535

    def __str__(self):
        return "[{}, {}, {}, {}]".format(self.min_x, self.min_y, self.max_x, self.max_y)

    __repr__ = __str__