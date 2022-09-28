import numpy as np 
import cv2 as cv

from typing import List 
from fern import Fern
from time import time 
from src.bounding_box import BoundingBox
from src.utils import calculate_covariance 
from utils import project_shape, similarity_transform

class FernCascade:

    _ferns: List[Fern]
    _second_level_num: int 

    def fit(
        self,
        images: List[np.ndarray[str]], # TODO: change type str to uchar(in c++). Find the same type in python
        current_shapes: List[np.ndarray[float]],
        ground_truth_shapes: List[np.ndarray[float]],
        bounding_box: List[BoundingBox],
        mean_shape: np.ndarray[float],
        second_level_num: int,
        candidate_pixel_num: int,
        fern_pixel_num: int,
        curr_level_num: int, 
        first_level_num: int
        ) -> List[np.ndarray[float]]:

        candidate_pixel_locations: np.ndarray[float] = np.zeros((candidate_pixel_num, 2), dtype=float)
        nearest_landmark_index: np.ndarray[int] = np.zeros((candidate_pixel_num, 1), dtype=int)
        regression_targets: List[np.ndarray[float]] = [np.ndarray([]) for _ in range(len(current_shapes))]
        self._second_level_num = second_level_num

        # calculate regression targets: the difference between ground truth shapes and current shapes
        # candidate_pixel_locations: the locations of candidate pixels, indexed relative to its nearest landmark on mean shape 

        for i in range(len(current_shapes)):
            regression_targets[i] = project_shape(ground_truth_shapes[i], bounding_box[i]) - project_shape(current_shapes[i], bounding_box[i])

            rotation: np.ndarray[float]
            scale: float 

            rotation, scale = similarity_transform(
                                mean_shape,
                                project_shape(current_shapes[i], bounding_box[i]),
                                rotation,
                                scale
                                )
            rotation = rotation.T 
            regression_targets[i] *= scale * rotation

        # get candidate pixel locations, please refer to 'shape-indexed features'
        for i in range(candidate_pixel_num):
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            if x**2 + y**2 > 1.0:
                i -= 1
                continue
            # find nearest landmark index 
            min_dist: float = 1e10
            min_index: int = 0
            for j in range(mean_shape.shape[0]):
                temp: float = np.pow(mean_shape[j, 0] - x, 2.) + np.pow(mean_shape[j, 1] - y, 2.)

                if temp < min_dist:
                    min_dist = temp 
                    min_index = j 
            
            candidate_pixel_locations[i, 0] = x - mean_shape[min_index, 0]
            candidate_pixel_locations[i, 1] = y - mean_shape[min_index, 1]
            nearest_landmark_index[i] = min_index 
        
        # get densities of candidate pixels for each image
        # for densities: each row is the pixel densities at each candidate pixels for an image 
        # Mat_<double> densities(images.size(), candidate_pixel_num);
        
        densities: List[List[float]] = [[] for _ in range(candidate_pixel_num)]
        for i in range(len(images)):
            rotation: np.ndarray[float]
            scale: float 
            temp: np.ndarray[float] = project_shape(current_shapes[i], bounding_box[i])

            rotation, scale = similarity_transform(temp, mean_shape, rotation, shape)
            for j in range(candidate_pixel_num):
                project_x = rotation[0, 0] * candidate_pixel_locations[j, 0] + rotation[0, 1] * candidate_pixel_locations[j, 1]
                project_y = rotation[1, 0] * candidate_pixel_locations[j, 0] + rotation[1, 1] * candidate_pixel_locations[j, 1]
                project_x *= scale * rotation
                project_y *= scale * rotation 
                index: int = nearest_landmark_index[j]
                real_x: int = project_x + current_shapes[i][index, 0]
                real_y: int = project_y + current_shapes[i][index, 1]
                real_x = np.max([0., real_x, images[i].shape[1] - 1.])
                real_y = np.max([0., real_y, images[i].shape[0] - 1.])
                densities[j].append(int(images[i][real_y, real_x]))
        
        # calculate the covariance between densities at each candidate pixels
        covariance: np.ndarray[float] = np.zeros((candidate_pixel_num, candidate_pixel_num))
        mean: np.ndarray[float]
        for i in range(candidate_pixel_num):
            for j in range(candidate_pixel_num):
                correlation_result: float = calculate_covariance(densities[i], densities)
                covariance[i, i] = correlation_result
                covariance[i, j] = correlation_result
        
        # train ferns 

        prediction: List[np.ndarray[float]] = [np.array([]) for _ in range(len(regression_targets))]
        for i in range(len(regression_targets)):
            prediction[i] = np.zeros((mean_shape.shape[0], 2, 1))

        self._ferns = [object() for _ in range(second_level_num)]
        start_fit = time()
        for i in range(second_level_num):
            temp: List[np.ndarray[float]] =  self._ferns[i].fit(
                                                densities,
                                                covariance,
                                                candidate_pixel_locations,
                                                nearest_landmark_index,
                                                regression_targets,
                                                fern_pixel_num
                                                )      
            # update regression targets
            for j in range(len(temp)):
                prediction[j] += temp[j]
                regression_targets[j] -= temp[j]
            if ((i + 1) % 50 == 0):
                print(f"Fern cascades: {curr_level_num} out of {first_level_num}")
                print(f"Ferns: {i+1} out of {second_level_num}")
                remaining_level_num: float = (first_level_num - curr_level_num) * 500 + second_level_num - i
                time_remaining: float = 0.02 * float(time() - start_fit) * remaining_level_num
                print(f"Expected remaining time: {time_remaining / 60} min {time_remaining % 60} sec")
                start_fit = time()

        for i in range(len(prediction)):
            rotation: np.ndarray[float]
            scale: float 
            rotation, scale = similarity_transform(project_shape(current_shapes[i], bounding_box[i]), mean_shape, rotation, scale)
            rotation = rotation.T 
            prediction[i] *= scale * rotation
        
        return prediction


    def predict(
        self,
        image: np.ndarray[str],
        bounding_box: BoundingBox,
        mean_shape: np.ndarray[float],
        shape: np.ndarray[float]
        ) -> np.ndarray[float]:
        result: np.ndarray[float] = np.zeros((shape.shape[0], 2, 1))
        rotation: np.ndarray[float]
        scale: float 
        rotation, scale = similarity_transform(project_shape(shape, bounding_box), mean_shape, rotation, scale)
        for i in range(self._second_level_num):
            result += self._ferns[i].predict(
                                    image,
                                    shape,
                                    rotation,
                                    bounding_box,
                                    scale
                                    )
        rotation, scale = similarity_transform(project_shape(shape, bounding_box), mean_shape, rotation, scale)
        rotation = rotation.T 
        result *= scale * rotation 
        
        return result 

