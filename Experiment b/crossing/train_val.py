import cv2 as cv

ori_img = cv.imread('image_original/train1.jpg')
print(ori_img.shape)
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 4), int(ori_img.shape[0] / 4)))
cv.imwrite('image_train/HR_crossing_train_0001.png', ori_img)
cv.imwrite('image_train/LR_crossing_train_0001.png', LR_img)

ori_img = cv.imread('image_original/train2.jpg')
print(ori_img.shape)
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 4), int(ori_img.shape[0] / 4)))
cv.imwrite('image_train/HR_crossing_train_0002.png', ori_img)
cv.imwrite('image_train/LR_crossing_train_0002.png', LR_img)

ori_img = cv.imread('image_original/val1.jpg')
print(ori_img.shape)
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 4), int(ori_img.shape[0] / 4)))
cv.imwrite('image_val/HR_crossing_val_0001.png', ori_img)
cv.imwrite('image_val/LR_crossing_val_0001.png', LR_img)