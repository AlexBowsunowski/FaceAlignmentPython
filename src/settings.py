from pydantic import BaseSettings 


class Settings(BaseSettings):
    TRAIN_IMAGE_PATH: str = "dataset/trainImages/"
    TRAIN_BOX_FILENAME: str = "dataset/boundingbox.txt"
    TRAIN_SHAPE_FILENAME: str = "dataset/keypoints.txt"
    TEST_IMAGE_PATH: str = "dataset/testImages/"
    TEST_BOX_FILENAME: str = "dataset/boundingbox_test.txt"
    MODEL_PATH: str = "models/"
    RESULTS_PATH: str = "results/"

    TRAIN_IMAGE_NUM: int = 1345
    TEST_IMAGE_NUM: int = 507

    TRAIN_CANDIDATE_PIXEL_NUM: int = 400
    TRAIN_FERN_PIXEL_NUM: int = 5
    TRAIN_FIRST_LEVEL_NUM: int = 10
    TRAIN_SECOND_LEVEL_NUM: int = 500

    LANDMARK_NUM: int = 29
    INITIAL_NUMBER: int = 20
