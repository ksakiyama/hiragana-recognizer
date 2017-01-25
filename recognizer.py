# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import nets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imglist', help='text file', default='testdata.txt')
    parser.add_argument('--initmodel', default='result/model_epoch_500',
                        help='Initialize the model from given file')
    parser.add_argument('--outfile', '-o', default='output.csv',
                        help='output text file name')
    parser.add_argument('--arch', '-a', default='cnn',
                        help='Network architecture')
    args = parser.parse_args()

    archs = {
        'cnn': nets.CNNSample,
        'cnnbn': nets.CNNSampleBN,
    }

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
    model = model = archs[args.arch]()
    model.train = False
    model.predict = True
    chainer.serializers.load_npz(args.initmodel, model)

    counter_wrong = 0

    for image, true_index in images_lndexes:
        # read image
        cvimg = cv2.imread(image, 0)
        cvimg = cv2.resize(cvimg, (32, 32))
        cvimg = cvimg / 255
        cvimg = cvimg.astype(np.float32)
        cvimg = cvimg.reshape(1, 32, 32)

        x = chainer.Variable(np.array([cvimg]))

        # ニューラルネットワークに推論させる
        ret = model(x).data[0]

        # 高確率な上位３候補を出すためソート
        rets = zip(range(0, 71), ret)
        rets = sorted(rets, key=lambda x: x[1], reverse=True)

        # 正解していたらスキップ
        if rets[0][0] == true_index:
            continue

        # 間違えてしまった内容を表示
        counter_wrong += 1
        print('====================')
        print("file:{}".format(image))
        print("label:{}".format(labeldic[true_index]))

        for idx, prob in rets[0:3]:
            print("{:02d}:{}:{}".format(idx, labeldic[idx], prob))

    print('====================')
    print('correct:{}'.format(len(images_lndexes) - counter_wrong))
    print('wrong:{}'.format(counter_wrong))
    print('accuracy:{:.3f}%'.format(1.0 - counter_wrong / len(images_lndexes)))


if __name__ == '__main__':
    main()
