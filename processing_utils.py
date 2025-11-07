import cv2

def normalize_image(img_path):
    img_weighted = cv2.imread(img_path)
    rows, cols = img_weighted.shape[:2]
    for i in range(rows):
        for j in range(cols):
            gray = 0.2989 * img_weighted[i,j][2] + 0.5870 * img_weighted[i,j][1] + 0.1140 * img_weighted[i,j][0]
            img_weighted[i,j] = [gray, gray, gray]
    img_blured = cv2.GaussianBlur(img_weighted, (5,5), 0)
    img_normalized = cv2.normalize(img_blured, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_normalized

def threshold_image(img, threshold):
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def morfological(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)
    return closed_img
