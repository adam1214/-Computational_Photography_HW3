import cv2 as cv

ori_img = cv.imread('image_original/train1.JPG')
print(ori_img.shape)
HR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 2), int(ori_img.shape[0] / 2)))
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 8), int(ori_img.shape[0] / 8)))
cv.imwrite('image_train/HR_book_train_0001.png', HR_img)
cv.imwrite('image_train/LR_book_train_0001.png', LR_img)

ori_img = cv.imread('image_original/train2.JPG')
print(ori_img.shape)
HR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 2), int(ori_img.shape[0] / 2)))
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 8), int(ori_img.shape[0] / 8)))
cv.imwrite('image_train/HR_book_train_0002.png', HR_img)
cv.imwrite('image_train/LR_book_train_0002.png', LR_img)

ori_img = cv.imread('image_original/val1.JPG')
print(ori_img.shape)
HR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 2), int(ori_img.shape[0] / 2)))
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 8), int(ori_img.shape[0] / 8)))
cv.imwrite('image_val/HR_book_val_0001.png', HR_img)
cv.imwrite('image_val/LR_book_val_0001.png', LR_img)

ori_img = cv.imread('image_original/LR_book.JPG')
print(ori_img.shape)
HR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 2), int(ori_img.shape[0] / 2)))
LR_img = cv.resize(ori_img, (int(ori_img.shape[1] / 8), int(ori_img.shape[0] / 8)))
cv.imwrite('reference/HR_book.png', HR_img)
cv.imwrite('image_test/LR_book.png', LR_img)