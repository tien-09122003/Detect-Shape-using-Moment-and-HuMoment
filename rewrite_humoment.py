import cv2
import numpy as np

def calculate_moments(image):
    moments = {
        'm00': 0.0, 'm10': 0.0, 'm01': 0.0, 'm11': 0.0,
        'm20': 0.0, 'm02': 0.0
    }
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x]
            moments['m00'] += pixel
            moments['m10'] += x * pixel
            moments['m01'] += y * pixel
            moments['m11'] += x * y * pixel
            moments['m20'] += x**2 * pixel
            moments['m02'] += y**2 * pixel
    
    return moments

def calculate_centroid(moments):
    if moments['m00'] != 0:
        x_centroid = moments['m10'] / moments['m00']
        y_centroid = moments['m01'] / moments['m00']
    else:
        x_centroid = 0
        y_centroid = 0
    return x_centroid, y_centroid

def calculate_central_moments(image, moments, x_centroid, y_centroid):
    central_moments = {
        'mu20': 0.0, 'mu02': 0.0
    }

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x]
            x_diff = x - x_centroid
            y_diff = y - y_centroid
            central_moments['mu20'] += (x_diff**2) * pixel
            central_moments['mu02'] += (y_diff**2) * pixel
    
    return central_moments

def calculate_normalized_central_moments(central_moments, moments):
    mu00 = moments['m00']
    if mu00 != 0:
        eta20 = central_moments['mu20'] / (mu00**2)
        eta02 = central_moments['mu02'] / (mu00**2)
    else:
        eta20 = 0
        eta02 = 0
    return eta20, eta02

def calculate_hu_moment_1(eta20, eta02):
    return eta20 + eta02

def main(image_path):
    
    # Đọc ảnh và chuyển thành mảng numpy (ảnh nhị phân)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Chuyển ảnh thành nhị phân
    _, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

    moments = calculate_moments(image)
    
    x_centroid, y_centroid = calculate_centroid(moments)
    
    central_moments = calculate_central_moments(image, moments, x_centroid, y_centroid)
    
    eta20, eta02 = calculate_normalized_central_moments(central_moments, moments)
    
    hu_moment_1 = calculate_hu_moment_1(eta20, eta02)
    
    if 0.000625 < hu_moment_1 < 0.000650:
        print("Circle")
    elif 0.000650 < hu_moment_1 < 0.000740:
        print("Square")
    elif 0.000760 < hu_moment_1 < 0.000854:
        print( "Triangle")
    else:
        print( "Unknown")
    
    print(f'Hu Moment 1 (manual): {hu_moment_1}')

    # So sánh với OpenCV
    opencv_moments = cv2.moments(image)
    opencv_hu_moments = cv2.HuMoments(opencv_moments).flatten()
    print(f'Hu Moment 1 (OpenCV): {opencv_hu_moments[0]}')

if __name__ == "__main__":
    image_path = r'test.png'
    main(image_path)
