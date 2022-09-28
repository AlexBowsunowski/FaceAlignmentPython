from statistics import covariance
import numpy as np 
import cv2

from typing import List, Tuple
from .bounding_box import BoundingBox


def project_shape(
    shape: np.ndarray[float],
    bounding_box: BoundingBox
    ) -> np.ndarray[float]:

    temp: np.ndarray[float] = np.zeros((shape.shape[0], 2))
    for i in range(shape.shape[0]):
        temp[i, 0] = (shape[i, 0] - bounding_box.centroid_x) / (bounding_box.width / 2.)
        temp[i, 1] = (shape[i, 1] - bounding_box.centroid_y) / (bounding_box.height / 2.)
    return temp


def re_project_shape(
    shape: np.ndarray[float],
    bounding_box: BoundingBox
    ) -> np.ndarray[float]:
    temp: np.ndarray[float] = np.zeros((shape.shape[0], 2))
    for i in range(shape.shape[0]):
        temp[i, 0] = shape[i, 0] * bounding_box.width / 2. + bounding_box.centroid_x
        temp[i, 1] = shape[i, 1] * bounding_box.height / 2. + bounding_box.centroid_y
    return temp   
    

def get_mean_shape(
    shapes: List[np.ndarray[float]],
    bounding_box: List[np.ndarray[BoundingBox]]
    ) -> float:

    result: np.ndarray[float] = np.zeros((shapes[0].shape[0], 1))
    for i in range(len(shapes)):
        result += project_shape(shapes[i], bounding_box[i])
    result *= 1. / len(shapes)

    return result 


def similarity_transform(
    shape1: np.ndarray[float],
    shape2: np.ndarray[float],
    rotation: np.ndarray[float],
    scale: float,
    ) -> Tuple[np.ndarray[float], float]:
    rotation = np.zeros((2, 2, 1))
    scale = 0 

    center_x_1 = 0
    center_y_1 = 0
    center_x_2 = 0
    center_y_2 = 0
    for i in range(shape1.shape[0]):
        center_x_1 += shape1[i, 0]
        center_y_1 += shape1[i, 1]
        center_x_2 += shape2[i, 0]
        center_y_2 += shape2[i, 1]

    center_x_1 /= shape1.shape[0]
    center_y_1 /= shape1.shape[0]
    center_x_2 /= shape2.shape[0]
    center_y_2 /= shape2.shape[0]
    
    temp1: np.ndarray[float] = shape1.copy()
    temp2: np.ndarray[float] = shape2.copy()
    for i in range(shape1.shape[0]):
        temp1[i, 0] -= center_x_1 
        temp1[i, 1] -= center_y_1
        temp2[i, 0] -= center_x_2 
        temp2[i, 1] -= center_y_2
    
    covariance1: np.ndarray[float]
    covariance2: np.ndarray[float]
    mean1: np.ndarray[float]
    mean2: np.ndarray[float]

    covariance1, mean1 = cv2.calcCovarMatrix(temp1, cv2.cv.CV_COVAR_COLS)
    covariance2, mean2 = cv2.calcCovarMatrix(temp2, cv2.cv.CV_COVAR_COLS)
    
    s1 = np.sqrt(np.linalg.norm(covariance1))
    s2 = np.sqrt(np.linalg.norm(covariance2))
    scale = s1 / s2 
    temp1 *= 1. / s1 
    temp2 *= 1. / s2 

    num = 0 
    den = 0
    for i in range(shape1.shape[0]):
        num += temp1[i, 1] * temp2[i, 0] - temp1[i, 0] * temp2[i, 1]
        den += temp1[i, 0] * temp2[i, 0] - temp1[i, 1] * temp2[i, 1]
    
    norm = np.sqrt(num**2 + den**2)
    sin_theta = num / norm 
    cos_theta = den / norm 
    rotation[0, 0] = cos_theta 
    rotation[0, 1] = -sin_theta 
    rotation[1, 0] = sin_theta 
    rotation[1, 1] = cos_theta

    return rotation, scale

def calculate_covariance(
    v_1: List[float],
    v_2: List[float]
    ) -> float:
    v1: np.ndarray[float] = np.array(v_1)
    v2: np.ndarray[float] = np.array(v_2)
    mean_1: float = np.mean(v1)[0]
    mean_2: float = np.mean(v2)[0]
    v1 -= mean_1 
    v2 -= mean_2 
    return np.mean(v1*v2)