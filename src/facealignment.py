from typing import List 

import cv2 
import numpy as np 
import numpy.typing as npt 

from bounding_box import BoundingBox
from settings import Settings
from shape_regressor import ShapeRegressor


class FaceAlignment:
    
    def __init__(self, name: str, settings: Settings):
        self.name = name 
        self.settings = settings 
        self.model = ShapeRegressor()

    
    def load_images(self, data_path: str, img_num: int):
        images = []
        for i in range(img_num):
            images.append(cv2.imread(f"{data_path}{str(i+1)}.jpg", 0))
        return images 


    def load_bounding_box(self, box_path: str):
        box_lines = []
        bounding_box = []
        with open(box_path) as f:
            box_lines = f.readlines()
        for line in box_lines:
            box_args = list(map(int, line.split()))
            bounding_box.append(
                BoundingBox(
                    start_x=box_args[0],
                    start_y=box_args[1],
                    width=box_args[2],
                    height=box_args[3]
                )
            )
        return bounding_box
    

    def load_shapes(self, shape_path: str):
        shapes_lines = []
        ground_truth_shapes = []
        with open(shape_path) as f:
            shapes_lines = f.readlines()
        for i in range(self.settings.TRAIN_IMAGE_NUM):
            shape_args = list(map(float, shapes_lines[i].split()))
            temp = np.zeros((self.settings.LANDMARK_NUM, 2))
            for j in range(self.settings.LANDMARK_NUM):
                temp[j, 0] = shape_args[j]
                temp[j, 1] = shape_args[self.settings.LANDMARK_NUM + j]
            ground_truth_shapes.append(temp)
        return ground_truth_shapes 
    

    def train(self):
        images = self.load_images(
            self.settings.TRAIN_IMAGE_PATH, self.settings.TRAIN_IMAGE_NUM
        )

        bounding_box = self.load_bounding_box(self.settings.TRAIN_BOX_FILENAME)

        ground_truth_shapes = self.load_shapes(self.settings.TRAIN_SHAPE_FILENAME)

        array_images = [np.asarray(image[:, :]) for image in images]
        self.model.train(
            array_images,
            ground_truth_shapes,
            bounding_box,
            self.settings.TRAIN_FIRST_LEVEL_NUM,
            self.settings.TRAIN_SECOND_LEVEL_NUM,
            self.settings.TRAIN_CANDIDATE_PIXEL_NUM,
            self.settings.TRAIN_FERN_PIXEL_NUM,
            self.settings.INITIAL_NUMBER
        )

    
    def test(self, manual: bool=False, saving_results: bool=False, showing_results: bool=False):
        images = self.load_images(
            self.settings.TEST_DATA_PATH, self.settings.TEST_IMAGE_NUM
        )

        bounding_box = self.load_bounding_box(self.settings.TEST_BOX_FILENAME)

        index = -1
        results = []

        while index < self.settings.TEST_IMAGE_NUM:
            if manual:
                index = int(input())
                index -= 1
            else:
                index += 1

            if index < 0 or index >= self.settings.TEST_IMAGE_NUM:
                break 

            current_shape = self.model.predict(
                np.asarray(images[index][:, :]),
                bounding_box[index],
                self.settings.INITIAL_NUMBER
            )

            test_image_1 = cv2.cvtColor(images[index].copy(), cv2.COLOR_GRAY2RGB)
            for i in range(self.settings.LANDMARK_NUM):
                test_image_1 = cv2.circle(
                    img=test_image_1,
                    center=(int(current_shape[i, 0]), int(current_shape[i, 1])),
                    radius=3,
                    color=(0, 255, 0),
                    thickness=-1,
                    lineType=8,
                    shift=0.
                )

            if saving_results:
                cv2.imwrite(
                    f"{self.settings.RESULTS_PATH}{self.name}_{str(index+1)}.jpg",
                    test_image_1,
                )

            if showing_results:
                cv2.imshow(f"Result for {str(index + 1)}.jpg", test_image_1)
                cv2.waitKey(0)
            results.append(current_shape)

    
    def load(self):
        with open(f"{self.settings.MODEL_PATH}{self.name}.txt") as inp:
            self.model.read(inp)
    

    def save(self):
        with open(f"{self.settings.MODEL_PATH}{self.name}.txt", "w") as out:
            self.model.write(out)
    
    
