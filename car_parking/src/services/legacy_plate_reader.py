import cv2
import numpy as np
import keras
from typing import Tuple
from .vehicle_detector import VehicleDetector
from pathlib import Path


path = str(Path(__file__).parent.parent) + "/models"

# dictionary of all classes so when model predicts number we can look up the digit or letter
CLASSES = {
    0: 0,   1: 1,   2: 2,
    3: 3,   4: 4,   5: 5,   
    6: 6,   7: 7,   8: 8,
    9: 9,   10: "A",    11: "B",
    12: "C",    13: "D",    14: "E",
    15: "F",    16: "G",    17: "H",
    18: "I",    19: "J",    20: "K",
    21: "L",    22: "M",    23: "N",    
    24: "O",    25: "P",    26: "Q",
    27: "R",    28: "S",    29: "T",
    30: "U",    31: "V",    32: "W",
    33: "X",    34: "Y",    35: "Z"
}


class PlatesReader():
    
    # loading text classifier model
    model = keras.models.load_model(path + r"/text_classifier.keras")
    vd = VehicleDetector()

    # method to get boxes from image where text is located
    async def get_rectangles(self, img_ori):
        # copy of img
        img_ori = img_ori
        # converting to gray
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
        gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
        # blur image
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        # threshold image
        img_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=3
        )

        contours, _= cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_NONE
        )

        contours_dict = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # insert to dict
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        MIN_AREA = 80 # min area of the box
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0

        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            # checking the area and ratio of every contour
            if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)
            
        MAX_DIAG_MULTIPLYER = 5
        MAX_ANGLE_DIFF = 12.0 
        MAX_AREA_DIFF = 0.5 
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 5 

        def find_chars(contour_list):
            matched_result_idx = []

            for d1 in contour_list:
                matched_contours_idx = []
                for d2 in contour_list:
                    if d1['idx'] == d2['idx']:
                        continue

                    dx = abs(d1['cx'] - d2['cx'])
                    dy = abs(d1['cy'] - d2['cy'])

                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                    distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                    if dx == 0:
                        angle_diff = 90
                    else:
                        angle_diff = np.degrees(np.arctan(dy / dx))
                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                    height_diff = abs(d1['h'] - d2['h']) / d1['h']

                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                        matched_contours_idx.append(d2['idx'])

                # append this contour
                matched_contours_idx.append(d1['idx'])

                if len(matched_contours_idx) < MIN_N_MATCHED:
                    continue

                matched_result_idx.append(matched_contours_idx)

                unmatched_contour_idx = []
                for d4 in contour_list:
                    if d4['idx'] not in matched_contours_idx:
                        unmatched_contour_idx.append(d4['idx'])

                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

                # recursive
                recursive_contour_list = find_chars(unmatched_contour)

                for idx in recursive_contour_list:
                    matched_result_idx.append(idx)

                break

            return matched_result_idx

        result_idx = find_chars(possible_contours)

        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(possible_contours, idx_list))
        return matched_result
    
    # method to crop image and leave only car
    async def get_cropped_img(self, img_ori):
        print("was called")
        # model to find car in the image 
        img = img_ori

        try:
            vehicle_boxes = self.vd.detect_vehicles(img)
        except Exception:
            return None
            
        areas = []
        for box in vehicle_boxes:
            x, y, w, h = box
            area = w * h
            areas.append(area)
        if len(areas) == 0:
            return None
        
        biggest_box = max(areas)
        idx = areas.index(biggest_box)
        x, y, w, h = vehicle_boxes[idx]
        img = img[y:y+h, x:x+w]
        return img

    # method to add padding to image
    async def resize_with_pad(self, image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
        
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape))/max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])

        if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
            ratio = float(min(new_shape)) / min(original_shape)
            new_size = tuple([int(x * ratio) for x in original_shape])

        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0] if new_shape[0] > new_size[0] else 0
        delta_h = new_shape[1] - new_size[1] if new_shape[1] > new_size[1] else 0
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return image

    # method to get prediction of text on the plate
    async def get_prediction(self, img):
        img_ori = img
        matched_result = await self.get_rectangles(img_ori)
        if len(matched_result) != 1:
            img_ori = await self.get_cropped_img(img_ori)
            if img_ori is None:
                return None
            matched_result = await self.get_rectangles(img_ori)
            if len(matched_result) != 1:
                return None
        matched_result = matched_result[0]
        
        idx_x = {}
        idx_x_sorted_dict = {}
        res = []

        for idx, d in enumerate(matched_result):
            idx_x[idx] = d['x']
        idx_x_sort = sorted(idx_x.items(), key=lambda x: x[1]) 

        for idx_x in idx_x_sort:
            idx_x_sorted_dict[idx_x[0]] = idx_x[1]

        for key in idx_x_sorted_dict.keys():
            res.append(matched_result[key])

        result = []
        new_shape = (24, 44)
        color = (255, 255, 255)
        for d in res:
            symbol = img_ori[d['y']:d['y']+d['h'], d['x']:d['x']+d['w']]
            img = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (15, 25))
            img = await self.resize_with_pad(img, new_shape, color)
            result.append(img)

        result = np.array(result)
        result = np.reshape(result, (result.shape[0], result.shape[1], result.shape[2], 1))
        result = result / 255.

        predicted = self.model.predict(result)
        predicted = [np.argmax(pred) for pred in predicted] 
        result = [str(CLASSES[pred]) for pred in predicted]
        result = ''.join(result)
        return result


pr = PlatesReader()