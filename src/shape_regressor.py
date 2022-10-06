import numpy as np
import cv2 as cv

from typing import List, IO
from fern_cascade import FernCascade
from src.bounding_box import BoundingBox
from src.utils import get_mean_shape, project_shape, re_project_shape 


class ShapeRegressor:

    def __init__(self):
        _first_level_num: int
        _landmark_num: int
        _fern_cascades: List[FernCascade]
        _mean_shape: np.ndarray[float]
        _training_shapes: List[np.ndarray[float]]
        _bounding_box: List[BoundingBox]


    def fit(
        self,
        images: List[np.ndarray[np.int_]],
        ground_truth_shapes: List[np.ndarray[float]], 
        bounding_box: List[BoundingBox],
        first_level_num: int,
        second_level_num: int,
        candidate_pixel_num: int,
        fern_pixel_num: int,
        initial_num: int
        ) -> None:

        print("Start fitting...")
        self._bounding_box = bounding_box
        self._training_shapes = ground_truth_shapes
        self._first_level_num = first_level_num
        self._landmark_num = ground_truth_shapes[0].shape[0]

        # data augmentation and multiple init
        augmented_images: List[np.ndarray[str]] # TODO: same problem 
        augmented_bounding_box: List[BoundingBox] 
        augmented_ground_truth_shapes: List[np.ndarray[float]]
        current_shapes: List[np.ndarray[float]]

        for i in range(len(images)):
            for j in range(initial_num):
                index: int = 0 
                index = int(np.random.uniform(0, len(images)))
                while index == i:
                    index = int(np.random.uniform(0, len(images)))
                augmented_images.append(images[i])
                augmented_ground_truth_shapes.append(ground_truth_shapes[i])
                augmented_bounding_box.append(bounding_box[i])
                # 1. Select ground truth shapes of other images as initial shapes
                # 2. Project current shape to bounding box of ground truth shapes
                temp: np.ndarray[float] = ground_truth_shapes[index]
                temp = project_shape(temp, bounding_box[i])
                temp = re_project_shape(temp, bounding_box[i])
                current_shapes.append(temp)

        # get mean shape from training shapes
        mean_shape = get_mean_shape(ground_truth_shapes, bounding_box)

        # train fern cascades 
        self._fern_cascades = [object() for _ in range(first_level_num)]
        prediction: List[np.ndarray[float]]
        for i in range(first_level_num):
            print(f"Training fern cascades: {i+1} out of {first_level_num}")
            prediction = self._fern_cascades[i].fit(
                                                augmented_images,
                                                current_shapes,
                                                augmented_ground_truth_shapes,
                                                augmented_bounding_box,
                                                mean_shape,
                                                second_level_num,
                                                candidate_pixel_num,
                                                fern_pixel_num,
                                                i+1,
                                                first_level_num
                                                )
            
            # update current shapes 

            for j in range(len(prediction)):
                current_shapes[j] = prediction[j] + project_shape(current_shapes[j], augmented_bounding_box[j])
                current_shapes[j] = re_project_shape(current_shapes[j], augmented_bounding_box[j])
            
    def predict(
        self,
        image: np.ndarray[str], # TODO: same problem
        bounding_box: BoundingBox,
        initial_num: int
        ) -> np.ndarray[float]:
        # generate multiple init 
        result: np.ndarray[float] = np.zeros((self, 2, 1))
        for i in range(initial_num):
            index = int(np.random.uniform(0, len(self._training_shapes)))
            current_shape: np.ndarray[float] = self._training_shapes[index]
            current_bounding_box: BoundingBox = self._bounding_box[index]
            current_shape = project_shape(current_shape, current_bounding_box)
            current_shape = re_project_shape(current_shape, bounding_box)
            for j in range(self._first_level_num):
                prediction: np.ndarray[float] = self._fern_cascades[j].predict(
                                                                        image,
                                                                        bounding_box,
                                                                        self._mean_shape,
                                                                        current_shape
                                                                    )
                current_shape = prediction + project_shape(current_shape, bounding_box)
                current_shape = re_project_shape(current_shape, bounding_box)
            result += current_shape
            
        return 1. / initial_num * result

    def write(self, out: IO):
        out.write(f"{self._first_level_num}\n")
        out.write(f"{self._mean_shape.shape[0]}\n")
        for i in range(self._landmark_num):
            out.write(f"{self._mean_shape[i, 0]} {self._mean_shape[i, 1]}")

        out.write("\n")
        out.write(f"{len(self._training_shapes)}\n")
        for i in range(len(self._training_shapes)):
            out.write(
                f"{int(self._bounding_box[i].start_x)} " 
                f"{int(self._bounding_box[i].start_y)} "
                f"{int(self._bounding_box[i].width)} "
                f"{int(self._bounding_box[i].height)} "
                f"{self._bounding_box[i].centroid_x} "
                f"{self._bounding_box[i].centroid_y} "
                )

            out.write("\n")
            for j in range(self._landmark_num):
                out.write(
                    f"{self._training_shapes[i][j, 0]} {self._training_shapes[i][j, 1]}"
                )
            out.write("\n")

        for fern_cascade in self._fern_cascades:
            fern_cascade.write(out)


    def read(self, inp: IO):
        self._first = int(inp.readline().strip())
        self._landmark_num = int(inp.readline().strip())
        self._mean_shape = np.zeros((self._landmark_num, 2))
        shapes = list(map(float, inp.readline().strip().split()))
        self._mean_shape[:, 0] = shapes[::2]
        self._mean_shape[:, 1] = shapes[1::2]

        training_num = int(inp.readline().strip())
        self._training_shapes = []
        self._bounding_box = []

        for i in range(training_num):
            box_args = list(map(float, inp.readline().strip().split()))
            self._bounding_box.append(
                BoundingBox(
                    start_x=box_args[0],
                    start_y=box_args[1],
                    width=box_args[2],
                    height=box_args[3]
                )
            )

            self._bounding_box[-1].centroid_x = box_args[4]
            self._bounding_box[-1].centroid_y = box_args[5]

            temp1 = np.zeros((self._landmark_num, 2))

            shapes = list(map(float, inp.readline().strip().split()))
            temp1[:, 0] = shapes[::2]
            temp1[:, 1] = shapes[1::2]
            self._training_shapes.append(temp1)

        self._fern_cascades = [FernCascade() for _ in range(self._first_level_num)]
        for fern_cascade in self._fern_cascades:
            fern_cascade.read(inp)


