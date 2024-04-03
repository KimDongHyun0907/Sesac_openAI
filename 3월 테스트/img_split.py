import cv2

image_paths = [["data/image_1.jpg", "data/image_2.jpg", "data/image_3.jpg", "data/image_4.jpg", "data/image_5.jpg"],
               ["data/image_6.jpg", "data/image_7.jpg", "data/image_8.jpg", "data/image_9.jpg", "data/image_10.jpg"], ["data/image_11.jpg"]]

cnt_x = 0
cnt_y = 0
num = 0
for img_list in image_paths:
    for img in img_list:
        image = cv2.imread("result.jpg")
        start_x, start_y, width, height = 10+cnt_x, 10+cnt_y, 150, 200
        end_x, end_y = start_x + width, start_y + height
        cropped_image = image[start_y:end_y, start_x:end_x]

        cv2.imshow("Cropped Image", cropped_image)
        cv2.imwrite(img, cropped_image)

        cnt_x += 150
    
    cnt_x = 0
    cnt_y += 250+num
    num += 10

cv2.waitKey(0)
cv2.destroyAllWindows()