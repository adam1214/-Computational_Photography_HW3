import cv2 as cv

ori_img = cv.imread('image_original/train1.jpg')
print(ori_img.shape)
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 4), int(ori_img.shape[0] / 4)))
cv.imwrite('image_train/HR_pups_train_0001.png', ori_img)
cv.imwrite('image_train/LR_pups_train_0001.png', LR_img)

ori_img = cv.imread('image_original/train2.jpg')
print(ori_img.shape)
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 4), int(ori_img.shape[0] / 4)))
cv.imwrite('image_train/HR_pups_train_0002.png', ori_img)
cv.imwrite('image_train/LR_pups_train_0002.png', LR_img)

ori_img = cv.imread('image_original/val1.jpg')
print(ori_img.shape)
HR_img = cv.resize(ori_img, (204, 244))
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 4), int(ori_img.shape[0] / 4)))
cv.imwrite('image_val/HR_pups_val_0001.png', HR_img)
cv.imwrite('image_val/LR_pups_val_0001.png', LR_img)