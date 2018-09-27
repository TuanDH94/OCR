import cv2
import json
from os import listdir
from os.path import isfile, isdir, join
import numpy as np

max_length_text = 128
alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
         'abcdefghijklmnopqrstuvwxyzÀáÂãÈÉÊÌÍÒóÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯư' \
         'ẠạẢảẤấẦầẨẩẪẫẬậẴắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ-/, '
def test():
    image = cv2.imread('data_sample/4.jpeg')
    image = cv2.resize(image, (1600, 250))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow('color_img', image)
    cv2.imshow('gray_img', gray_image)
    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary_img', im_bw)
    cv2.imwrite('gray_img.jpg', gray_image)
    cv2.imwrite('binary_img.jpg', im_bw)
    cv2.waitKey(0)  # Waits forever for user to press any key


def test2():
    file = open('data_sample/labels.json', mode='r', encoding='utf-8')
    json_data = json.load(file)
    print(json_data['1.jpg'])


def img_read_gray(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (1600, 250))
    return gray


def json_label_read(json_file):
    file = open(json_file, mode='r', encoding='utf-8')
    json_data = json.load(file)
    return json_data


# Translation of characters to unique integer values
def text_to_labels(alphabet, text):
    ret = []
    for i in range(max_length_text):
        if i < len(text) - 1:
            ret.append(alphabet.find(text[i]))
        else:
            ret.append(-1)
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(alphabet, labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def load_data():
    data_paths = 'data_sample'
    label_json_paths = 'data_sample/labels.json'
    # dest11 = 'E:\\Data\\train_voice\\debug\\'
    # img processing
    all_files = [f for f in listdir(data_paths) if isfile(join(data_paths, f))
                 and f.endswith(('.JPG', '.jpg', '.jpeg', '.JPEG'))]
    img_json_label = json_label_read(label_json_paths)
    X_img = []
    Y_label = []
    for file in all_files:
        img_path = join(data_paths, file)
        x_img = img_read_gray(img_path)
        X_img.append(x_img)
        text = img_json_label[file]
        label = text_to_labels(alphabet, text)
        Y_label.append(label)
    X_data = np.asarray(X_img)
    Y_label = np.asarray(Y_label)
    return X_data, Y_label

if __name__ == '__main__':
    X, Y = load_data()
