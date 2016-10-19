# -*- coding: utf-8 -*-
import os
import sys
import random

# 画像の保存先
srcdir = 'imgs'

# 学習に使う画像枚数
train_size = 9600


def save_text(textnm, images):
    with open(textnm, 'w') as f:
        for imgnm in sorted(images):
            imgpath = os.path.join(srcdir, imgnm)
            label = imgnm.split('_')[1]
            label = int(label)
            f.write("{} {}\n".format(imgpath, str(label)))


def main():
    # 画像のリストを取得
    # ランダムに並び替え、先頭からtrain_sizeを学習データにする
    images = os.listdir(srcdir)
    random.shuffle(images)
    trains = images[:train_size]
    tests = images[train_size:]

    save_text('traindata.txt', trains)
    save_text('testdata.txt', tests)


if __name__ == '__main__':
    main()
