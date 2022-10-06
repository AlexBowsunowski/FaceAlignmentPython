class BoundingBox:
    

    def __init__(
            self,
            start_x: float=0.,
            start_y: float=0.,
            width: float=0.,
            height: float=0.
        ):
        self.start_x = 0
        self.start_y = 0
        self.width = 0
        self.height = 0
        self.centroid_x = self.start_x + self.width / 2
        self.centroid_y = self.start_y + self.height / 2
