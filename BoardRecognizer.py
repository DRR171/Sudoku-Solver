"""
sudoku_image_reader.py

Converted into a BoardRecognizer class to be used from GUIs.

Public API:
    recognizer = BoardRecognizer(model_path='digit_classifier.h5', square_size=450, debug=False)
    board = recognizer.recognize('sudoku_photo.jpg')  # returns 9x9 numpy array with 0 for empty

Optional methods:
    recognizer.train_and_save(path='digit_classifier.h5', epochs=5)

Requires: opencv-python, numpy, tensorflow
"""

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models


class BoardRecognizer:
    def __init__(self, model_path=None, square_size=450, debug=False):
        """
        model_path: optional path to a Keras .h5 model for digit classification (0-9)
        square_size: size (pixels) of the warped squared board used for cell extraction
        debug: if True, show intermediate visualization windows
        """
        self.square_size = int(square_size)
        self.debug = bool(debug)
        self.model = None
        if model_path is not None:
            self.load_model(model_path)

    # ----------------------------- Model helpers -----------------------------
    def build_simple_cnn(self, input_shape=(28, 28, 1)):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_and_save(self, save_path='digit_classifier.h5', epochs=5):
        """Train a simple CNN on MNIST and save it. Good for quick starts but not ideal for messy photos."""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        model = self.build_simple_cnn()
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=128)
        model.save(save_path)
        self.model = model
        return model

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = tf.keras.models.load_model(path)
        return self.model

    # ----------------------------- Geometry / transform helpers -----------------------------
    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def _four_point_transform(self, image, pts):
        rect = self._order_points(pts.astype('float32'))
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [self.square_size - 1, 0],
            [self.square_size - 1, self.square_size - 1],
            [0, self.square_size - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.square_size, self.square_size))
        return warped

    # ----------------------------- Grid detection -----------------------------
    def _find_puzzle_contour(self, gray):
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        return None

    # ----------------------------- Digit extraction -----------------------------
    def _extract_digit(self, cell_img):
        # cell_img: grayscale square
        h, w = cell_img.shape
        margin = int(min(h, w) * 0.12)
        roi = cell_img[margin:h - margin, margin:w - margin]
        if roi.size == 0:
            return None

        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        (x, y, w_c, h_c) = cv2.boundingRect(c)
        area_ratio = cv2.countNonZero(mask) / float(roi.shape[0] * roi.shape[1])
        if area_ratio < 0.01:  # empty or noise
            return None

        digit = thresh[y:y + h_c, x:x + w_c]
        h_d, w_d = digit.shape
        if h_d == 0 or w_d == 0:
            return None

        # scale to fit within 20x20 box then center in 28x28
        scale = 20.0 / max(h_d, w_d)
        new_w = max(1, int(round(w_d * scale)))
        new_h = max(1, int(round(h_d * scale)))
        resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype="uint8")
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        out = canvas.astype('float32') / 255.0
        out = out.reshape(28, 28, 1)
        return out

    # ----------------------------- Main pipeline -----------------------------
    def recognize(self, image_path_or_array):
        """
        Accepts a path to an image file or a BGR numpy array. Returns a 9x9 numpy int array with 0 for empty cells.
        """
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        puzzle_cnt = self._find_puzzle_contour(gray)
        if puzzle_cnt is None:
            raise RuntimeError("Could not find Sudoku puzzle contour")

        warped = self._four_point_transform(orig, puzzle_cnt)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)

        side = warped_blur.shape[0]
        cell_side = side // 9

        board = np.zeros((9, 9), dtype=np.int32)

        for r in range(9):
            for c in range(9):
                y1 = r * cell_side
                y2 = (r + 1) * cell_side
                x1 = c * cell_side
                x2 = (c + 1) * cell_side
                cell = warped_blur[y1:y2, x1:x2]
                digit_in = self._extract_digit(cell)
                if digit_in is None:
                    board[r, c] = 0
                else:
                    if self.model is None:
                        board[r, c] = 0
                    else:
                        pred = self.model.predict(digit_in.reshape(1, 28, 28, 1))
                        label = int(np.argmax(pred, axis=1)[0])
                        board[r, c] = label

        if self.debug:
            vis = warped.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            for r in range(9):
                for c in range(9):
                    val = board[r, c]
                    if val != 0:
                        x = int((c + 0.5) * cell_side)
                        y = int((r + 0.65) * cell_side)
                        cv2.putText(vis, str(int(val)), (x - cell_side // 6, y), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('recognized', vis)
            cv2.waitKey(1)
            # Do not block; caller can destroy windows when needed

        return board


# ----------------------------- CLI usage -----------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BoardRecognizer CLI')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model', help='Path to digit classifier .h5 file', default=None)
    parser.add_argument('--train', action='store_true', help='Train a model on MNIST and save to --model')
    parser.add_argument('--debug', action='store_true', help='Show debug window')
    args = parser.parse_args()

    recognizer = BoardRecognizer(model_path=args.model, square_size=450, debug=args.debug)
    if args.train:
        recognizer.train_and_save(save_path=args.model or 'digit_classifier.h5', epochs=5)

    board = recognizer.recognize(args.image)
    print('Recognized board:')
    print(board)
