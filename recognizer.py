# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


# CNNを実装したもの
class CNNSample(chainer.Chain):

    def __init__(self):
        super(CNNSample, self).__init__(
            conv1=L.Convolution2D(None, 32, 3),
            conv2=L.Convolution2D(None, 32, 3),
            conv3=L.Convolution2D(None, 64, 3),
            conv4=L.Convolution2D(None, 64, 3),
            l1=L.Linear(None, 256),
            l2=L.Linear(None, 71),
        )
        self.train = False

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, train=self.train)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, train=self.train)

        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        h = F.relu(self.l2(h))
        return F.softmax(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imglist', help='text file', default='testdata.txt')
    parser.add_argument('--initmodel', default='result/model_epoch_500',
                        help='Initialize the model from given file')
    parser.add_argument('--outfile', '-o', default='output.csv',
                        help='output text file name')
    args = parser.parse_args()

    # Chainerのidxとひらがなの辞書
    labeldic = {}
    with open('labels.txt', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            c = line.rstrip('\n')
            labeldic[int(idx)] = c

    # テスト画像のリストを取得
    images_lndexes = []
    with open(args.imglist, 'r') as f:
        for line in f.readlines():
            imgpath, labelidx = line.split(' ')
            labelidx = labelidx.rstrip('\n')
            # print("{}:{}".format(imgpath, labelidx))
            images_lndexes.append((imgpath, int(labelidx)))

    # モデルをロードする
    model = CNNSample()
    # model = L.Classifier(CNNSample())
    chainer.serializers.load_npz(args.initmodel, model)

    # 試しに１つ画像を読み込む
    for image, true_index in images_lndexes:
        # read image
        cvimg = cv2.imread(image, 0)
        cvimg = cv2.resize(cvimg, (32, 32))
        cvimg = cvimg / 255
        cvimg = cvimg.astype(np.float32)
        cvimg = cvimg.reshape(1, 32, 32)

        x = chainer.Variable(np.array([cvimg]))

        # predict
        ret = model(x).data[0]

        # calc top3
        rets = zip(range(0, 71), ret)
        rets = sorted(rets, key=lambda x: x[1], reverse=True)

        pred_idx, _ = rets[0]
        if pred_idx == true_index:
            continue

        print('====================')
        print("filename:{}".format(image))
        print("label:{}".format(labeldic[true_index]))

        for idx, prob in rets[0:3]:
            print("{}:{}:{}".format(idx, labeldic[idx], prob))


if __name__ == '__main__':
    main()
