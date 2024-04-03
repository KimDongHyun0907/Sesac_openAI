import cv2

# 이미지 파일 경로 리스트
image_paths = ["data/image_1.jpg", "data/image_2.jpg", "data/image_3.jpg", "data/image_4.jpg", "data/image_5.jpg",
               "data/image_6.jpg", "data/image_7.jpg", "data/image_8.jpg", "data/image_9.jpg", "data/image_10.jpg", "data/image_11.jpg"]


# 이미지 불러오기 및 가로로 병합
images = []
for image_path in image_paths:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is not None:
        images.append(img)

if images:  # 이미지가 있는 경우에만 병합
    merged_image = cv2.hconcat(images)

    # 병합된 이미지 보여주기
    cv2.imshow("Merged Image", merged_image)
    cv2.imwrite('merged_image.jpg', merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("이미지를 불러올 수 없습니다.")
