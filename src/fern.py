import numpy as np
import cv2 as cv
from typing import List
from src.bounding_box import BoundingBox
from utils import calculate_covariance

class Fern:

    _fern_pixel_num: int 
    _landmark_num: int 
    _selected_nearest_landmark_index: np.ndarray[int]
    _threshold: np.ndarray[float] 
    _selected_pixel_locations: np.ndarray[int]
    _selected_pixel_index: np.ndarray[int]
    _bin_output: List[np.ndarray[float]]


    def fit(self, candidate_pixel_intensity: List[List[float]], 
            covariance: np.ndarray[float],
            candidate_pixel_locations: np.ndarray[float],
            nearest_landmark_index: np.ndarray[int],
            regression_targets: List[np.ndarray[float]],
            fern_pixel_num: int) -> List[np.ndarray[float]]:
            
            self._fern_pixel_num = fern_pixel_num
            self._landmark_num = regression_targets[0].shape[0]
            self._selected_pixel_index = np.zeros((self._fern_pixel_num, 2))
            self._selected_nearest_landmark_index = np.zeros((self._fern_pixel_num, 4))
            candidate_pixel_num: int = candidate_pixel_locations.shape[0]

            self._threshold = np.zeros((fern_pixel_num, 1))
            
            for i in range(fern_pixel_num):
                random_direction: np.ndarray = np.random.uniform(
                    low=-1.1,
                    high=1.1,
                    size=(self._landmark_num, 2))
                
                random_direction = cv.normalize(random_direction, random_direction)
                projection_result: List[float] = [0] * len(regression_targets)
                # project regression targets along the random direction
                for j in range(len(regression_targets)):
                    temp: float = np.sum(regression_targets[j]*random_direction, axis=1)[0]
                    projection_result[j] = temp 
                
                covariance_projection_density: np.ndarray[float] = np.zeros((candidate_pixel_num, 1))
                for j in range(candidate_pixel_num):
                    covariance_projection_density[j] = calculate_covariance(projection_result, candidate_pixel_intensity[j])
                

                # find max correlation 

                max_correlation: float = -1
                max_pixel_index_1: int = 0
                max_pixel_index_2: int = 0

                for j in range(candidate_pixel_num):
                    for k in range(candidate_pixel_num):
                        temp1 = covariance[j, j] + covariance[k, k] - 2 * covariance[j, k]
                        if abs(temp1) < 1e-10:
                            continue
                        flag: bool = False 

                        for p in range(i):
                            if j == self._selected_pixel_index[p, 0] and k == self._selected_pixel_index[p, 1]:
                                flag = True 
                                break 
                            elif j == self._selected_pixel_index[p, 1] and k == self._selected_pixel_index[p, 0]:
                                flag = True
                                break 
                        if flag:
                            continue
                        temp: float = (covariance_projection_density[j] - covariance_projection_density[k]) / np.sqrt(temp1)
                        if abs(temp) > max_correlation:
                            max_correlation = temp 
                            max_pixel_index_1 = j 
                            max_pixel_index_2 = k 
                
                self._selected_pixel_index[i, 0] = max_pixel_index_1
                self._selected_pixel_index[i, 1] = max_pixel_index_2
                self._selected_pixel_locations[i, 0] = candidate_pixel_locations[max_pixel_index_1, 0]
                self._selected_pixel_locations[i, 1] = candidate_pixel_locations[max_pixel_index_1, 1]
                self._selected_pixel_locations[i, 2] = candidate_pixel_locations[max_pixel_index_2, 0]
                self._selected_pixel_locations[i, 3] = candidate_pixel_locations[max_pixel_index_2, 1]


                max_diff: float = -1 
                for j in range(len(candidate_pixel_intensity[max_pixel_index_1])):
                    temp: float = candidate_pixel_intensity[max_pixel_index_1][j] - candidate_pixel_intensity[max_pixel_index_2][j]
                    if abs(temp) > max_diff:
                        max_diff = abs(temp)
                
                self._threshold[i] = np.random.uniform(-0.2*max_diff, 0.2*max_diff)
            
            bin_num: int = np.pow(2.0, fern_pixel_num)
            shapes_in_bin: List[List[int]] = [[] * bin_num]
            for i in range(len(regression_targets)):
                index: int = 0
                for j in range(fern_pixel_num):
                    density_1: float = candidate_pixel_intensity[self._selected_pixel_index[j, 0]][i]
                    density_2: float = candidate_pixel_intensity[self._selected_pixel_index[j, 1]][i]
                    if density_1 - density_2 >= self._threshold[j]:
                        index += np.pow(2.0, j)
                shapes_in_bin[index].append(i)
            
            # get bin output 

            prediction: List[np.ndarray[float]] = [np.array([]) * len(regression_targets)] 
            self._bin_output = [np.array([]) * bin_num]
            for i in range(bin_num):
                temp: np.ndarray[float] = cv.zeros((self._landmark_num, 2, 1))
                bin_size: int = len(shapes_in_bin[i])
                for j in range(bin_size):
                    index = shapes_in_bin[i][j]
                    temp += regression_targets[index]
                if bin_size == 0:
                    self._bin_output[i] = temp 
                    continue
                temp = (1. / ((1. + 1000. / bin_size) * bin_size)) * temp 
                self._bin_output[i] = temp 
                for j in range(bin_size):
                    index = shapes_in_bin[i][j]
                    prediction[index] = temp 
            
            return prediction 


    def predict(
        self,
        image: np.ndarray[str], # TODO: change type str to uchar(in c++). Find the same type in python
        shape: np.ndarray[float],
        rotation: np.ndarray[float],
        bounding_box: BoundingBox,
        scale: float
        ) -> np.ndarray[float]:

        index = 0
        for i in range(self._fern_pixel_num):
            nearest_landmark_index_1: int = self._selected_nearest_landmark_index[i, 0]
            nearest_landmark_index_2: int = self._selected_nearest_landmark_index[i, 1]
            x: int = self._selected_pixel_locations[i, 0]
            y: int = self._selected_pixel_locations[i, 1]
            project_x: float = scale * (rotation[0, 0] * x + rotation[0, 1] * y) * bounding_box.width / 2. + shape[nearest_landmark_index_1, 0]
            project_y: float = scale * (rotation[1, 0] * x + rotation[1, 1] * y) * bounding_box.height / 2. + shape[nearest_landmark_index_1, 1]

            project_x: int = int(np.max(0, np.min(project_x, image.shape[1] - 1.)))
            project_y: int = int(np.max(0, np.min(project_y, image.shape[0] - 1.)))
            intensity_1: float = image[project_y, project_x]

            x = self._selected_pixel_locations[i, 2]
            y = self._selected_pixel_locations[i, 3]
            project_x: float = scale * (rotation[0, 0] * x + rotation[0, 1] * y) * bounding_box.width / 2. + shape[nearest_landmark_index_2, 0]
            project_y: float = scale * (rotation[1, 0] * x + rotation[1, 1] * y) * bounding_box.height / 2. + shape[nearest_landmark_index_2, 1]

            project_x: int = int(np.max(0, np.min(project_x, image.shape[1] - 1.)))
            project_y: int = int(np.max(0, np.min(project_y, image.shape[0] - 1.)))
            intensity_2: float = image[project_y, project_x]
            
            if intensity_1 - intensity_2 >= self._threshold[i]:
                index += int(np.pow(2, i))
        return self._bin_output[index]