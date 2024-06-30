import cv2
import numpy as np
import os

def calculate_hu_moments(img):
    moments = cv2.moments(img)
    huMoments = cv2.HuMoments(moments)
    return huMoments.flatten()

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def collect_hu_moments(data_path):
    shape_hu_moments = {
        "Circle": [], 
        "Square": [],
        "Triangle": [],
    }

    for shape in shape_hu_moments.keys():
        shape_path = os.path.join(data_path, shape)
        for file in os.listdir(shape_path):
            img_path = os.path.join(shape_path, file)
            img = load_image(img_path)
            _, binary_image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
            hu_moments = calculate_hu_moments(binary_image)
            shape_hu_moments[shape].append(hu_moments[0])
    
    return shape_hu_moments

def analyze_hu_moments(shape_hu_moments):
    thresholds = {}
    for shape, hu_values in shape_hu_moments.items():
        min_val = np.min(hu_values)
        max_val = np.max(hu_values)
        mean_val = np.mean(hu_values)
        thresholds[shape] = (min_val, max_val, mean_val)
    return thresholds

def main():
    
    
    data_path = "train_humoment"
    print(f'Tải metadata từ: {data_path}')

    # Bước 1: Thu thập và tính toán Hu Moments
    shape_hu_moments = collect_hu_moments(data_path)

    # Bước 2: Phân tích Hu Moments
    thresholds = analyze_hu_moments(shape_hu_moments)
    for shape, (min_val, max_val, mean_val) in thresholds.items():
        print(f"{shape} - Min: {min_val}, Max: {max_val}, Mean: {mean_val}")

if __name__ == "__main__":
    main()
