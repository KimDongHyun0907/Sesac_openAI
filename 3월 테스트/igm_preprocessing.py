import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('coupon.jpg', cv2.IMREAD_COLOR)
img180 = cv2.rotate(img, cv2.ROTATE_180)

kernel_sharpen = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])

sharp_image = cv2.filter2D(img180, -1, kernel_sharpen)

# CLAHE를 적용하여 명암 대비를 향상시킵니다.
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

# BGR 채널을 분리합니다.
b, g, r = cv2.split(sharp_image)

# 각 채널에 대해 CLAHE를 적용합니다.
enhanced_b = clahe.apply(b)
enhanced_g = clahe.apply(g)
enhanced_r = clahe.apply(r)

# CLAHE를 적용한 각 채널을 병합합니다.
enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))

# 결과 이미지 저장
cv2.imwrite('result.jpg', enhanced_img)

# 결과 이미지 보여주기
cv2.imshow('result', enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()