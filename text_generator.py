# -*- coding: utf-8 -*-
import os
import sys
import random

# 画像の保存先
srcdir = 'imgs'


def fild_all_files(directory):
    for root, dirs, files in os.walk(directory):
        if not os.path.isdir(root):
            yield root
        for file in files:
            yield os.path.join(root, file)


def save_text(textnm, images):
    with open(textnm, 'w') as f:
        for imgnm in sorted(images):
            label = imgnm.split('_')[1]
            label = int(label)
            f.write("{} {}\n".format(imgnm, str(label)))


def main():
    # 画像のリストを取得
    # ランダムに並び替え、先頭からtrain_sizeを学習データにする
    images = list(fild_all_files(srcdir))

    train_size = int(len(images) * 0.8)
    random.shuffle(images)
    trains = images[:train_size]
    tests = images[train_size:]

    save_text('traindata.txt', trains)
    save_text('testdata.txt', tests)


if __name__ == '__main__':
    main()
