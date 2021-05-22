import cv2 as cv

for i in range(3,11,1):
    ori_img = cv.imread('train'+str(i)+'.JPG')
    shape_1 = ori_img.shape[1]
    shape_0 = ori_img.shape[0]
    while True:
        if shape_1 % 4 == 0:
            break
        else:
            shape_1 -= 1
    while True:
        if shape_0 % 4 == 0:
            break
        else:
            shape_0 -= 1
    HR_img = cv.resize(ori_img, (int(shape_1), int(shape_0)))
    LR_img = cv.resize(ori_img, (int(shape_1 / 4), int(shape_0 / 4)))
    if i == 10:
        cv.imwrite('../image_train/HR_zebra_train_00'+str(i)+'.png', ori_img)
        cv.imwrite('../image_train/LR_zebra_train_00'+str(i)+'.png', LR_img)
    else:
        cv.imwrite('../image_train/HR_zebra_train_000'+str(i)+'.png', ori_img)
        cv.imwrite('../image_train/LR_zebra_train_000'+str(i)+'.png', LR_img)

for i in range(2,6,1):
    ori_img = cv.imread('val'+str(i)+'.JPG')
    shape_1 = ori_img.shape[1]
    shape_0 = ori_img.shape[0]
    while True:
        if shape_1 % 4 == 0:
            break
        else:
            shape_1 -= 1
    while True:
        if shape_0 % 4 == 0:
            break
        else:
            shape_0 -= 1
    HR_img = cv.resize(ori_img, (int(shape_1), int(shape_0)))
    LR_img = cv.resize(ori_img, (int(shape_1 / 4), int(shape_0 / 4)))
    cv.imwrite('../image_val/HR_zebra_val_000'+str(i)+'.png', HR_img)
    cv.imwrite('../image_val/LR_zebra_val_000'+str(i)+'.png', LR_img)
