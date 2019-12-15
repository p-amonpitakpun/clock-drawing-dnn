from preprocess import preprocess, get_image
import numpy as np
import cv2


shape = (500, 500)
dat = preprocess('data\\raw\\CD_PD.mat')
image_dir = f'data\\image-{shape[0]}x{shape[1]}\\'

print('write images to dir', image_dir)
for n in range(dat.shape[0]):
    x = dat[n]['x']
    y = dat[n]['y']

    array = get_image(x, y, shape=shape)

    img = np.zeros((shape[0], shape[1], 3), np.float32)

    for j in range(shape[0]):
        for i in range(shape[1]):
            if array[i][j]:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow('image', img)
    cv2.imwrite(image_dir + 'clock' + '0'*(3 - len(str(n))) + str(n) + '.jpg', img)
    cv2.waitKey(10)

    print('write image', '0'*(3 - len(str(n))) + str(n), '\tshape', shape)
