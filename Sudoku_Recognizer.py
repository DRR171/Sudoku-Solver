import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# import time

class Sudoku_Recognizer:
    
    def __init__(self):
        self.image_path = None
        self.board_image = None
        self.digit_recog_model = None
        self.image_list = None
        self.recognized_board = None
        
    def set_image_path(self, img_path):
        self.image_path = img_path
    
    def get_recognized_board(self):
        return self.recognized_board
    
    def load_digit_model(self):
        try:
            # self.digit_recog_model = load_model('digit_recognizer_model.h5', compile=False)
            # self.digit_recog_model = load_model('digit_recognizer_2025_09_02.h5', compile=False)
            # self.digit_recog_model = load_model('digit_recognizer_2025_09_03.h5', compile=False)
            # self.digit_recog_model = load_model('digit_recognizer_2025_09_03_21_41.h5', compile=False)
            # self.digit_recog_model = load_model('digit_recognizer_2025_09_03_22_43.h5', compile=False)
            self.digit_recog_model = load_model('digit_recognizer_2025_09_04_11_11.h5', compile=False)
        except Exception as e:
            return e
    
    def read_image_file(self):
        try:
            self.board_image = cv2.imread(self.image_path)
        except Exception as e:
            return e
    
    def read_image_crop(self, img):
        
        img_np = np.array(img)
        try:
            self.board_image = img_np.astype(np.uint8)
        except Exception as e:
            return e    
    
    def create_cell_image_list(self):
        gray_img = cv2.cvtColor(self.board_image, cv2.COLOR_BGR2GRAY)
        
        
        corners1 = cv2.goodFeaturesToTrack(gray_img,
                                          maxCorners=200,
                                          qualityLevel=0.01,
                                          minDistance=1)
        
        corners2 = cv2.goodFeaturesToTrack(gray_img,
                                          maxCorners=200,
                                          qualityLevel=0.05,
                                          minDistance=20)
        
        corners3 = cv2.goodFeaturesToTrack(gray_img,
                                          maxCorners=200,
                                          qualityLevel=0.1,
                                          minDistance=10)
        
        corners = np.concatenate([corners1, corners2, corners3], axis=0)
        corners = np.int0(corners)
    
        # Extract corners as a list of (x, y)
        points = [tuple(c.ravel()) for c in corners]
    
        # Top-left = minimum x+y
        top_left = min(points, key=lambda p: p[0] + p[1])
    
        # Bottom-right = maximum x+y
        bottom_right = max(points, key=lambda p: p[0] + p[1])
    
        cell_x_size = int((bottom_right[0] - top_left[0])/9)
        cell_y_size = int((bottom_right[1] - top_left[1])/9)
        
        x0_tl = top_left[0]
        y0_tl = top_left[1]
    
        x0_br = bottom_right[0]
        y0_br = bottom_right[1]
    
        tl_list = []
        br_list = []
    
        for y_ind in range(1, 10):
            
            y_curr_tl = y0_tl + y_ind*cell_y_size
            y_curr_br = y0_br - y_ind*cell_y_size
            
            for x_ind in range(1, 10):
                x_curr_tl = x0_tl + x_ind*cell_x_size
                tl_list.append((x_curr_tl, y_curr_tl))
                
                x_curr_br = x0_br - x_ind*cell_x_size
                br_list.append((x_curr_br, y_curr_br))
                
        br_list.reverse()
        
        fr = 0
        ROI_list = []
        
        for ind in range(81):
            
            ROI = gray_img[br_list[ind][1]+fr:tl_list[ind][1]-fr,
                           br_list[ind][0]+fr:tl_list[ind][0]-fr].copy()
            
            resized_ROI = cv2.resize(ROI, (28, 28))
            resized_ROI = resized_ROI.astype(np.float32)
            gray_ROI = resized_ROI.astype(np.float32) / 255.0
            ROI_list.append(gray_ROI)
            
        self.image_list = np.array(ROI_list).reshape(-1, 28, 28, 1)    
    
    def recognize_board(self):
        pred_digit = self.digit_recog_model.predict(self.image_list, batch_size=81)
        self.recognized_board = np.argmax(pred_digit, axis=1).reshape((9, 9))
       
    def plot_digit_cells_preds(self):
        fig, axes = plt.subplots(9, 9, figsize=(12, 12))
        fig.suptitle("9x9 Grid on digits", fontsize=18)

        for i, ax in enumerate(axes.flat):
            ax.imshow(self.image_list[i], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Predicted digit: {self.recognized_board[i//9][i%9]}", fontsize=10)
            ax.axis('off')
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
