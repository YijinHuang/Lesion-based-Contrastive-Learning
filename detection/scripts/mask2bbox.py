# coding=utf-8
import os
import cv2
import glob as gb
from tqdm import tqdm


img_path = gb.glob("/dataset/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/2. Haemorrhages/*")
img_savepath = "./output/Haemorrhages"


def generate(img, path, csvfile, class_):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    contours, hierachy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), (255, 255, 255), 1)
        csvfile.write('{},{},{},{},{},{}\n'.format(
            path + '.jpg', x - 1, y - 1, x + w, y + h, class_
        ))
    return img


if __name__ == '__main__':
    with open('./output/bbox_Microaneurysms.csv', 'w', encoding='utf-8') as csvfile:
        for path in tqdm(img_path):
            (img_dir, temp_filename) = os.path.split(path)
            (f_name, fe_name) = os.path.splitext(path)
            img = cv2.imread(path)
            img = generate(img, f_name, csvfile, 'mic')
            savepath = os.path.join(img_savepath, temp_filename)
            cv2.imwrite(savepath, img)
