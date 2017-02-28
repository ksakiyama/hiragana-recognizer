import os
import sys
import argparse
from PIL import Image
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import nets


def main():
    # ここで実行時の引数を設定できます
    # 例：python inferencer.py [テストデータ] --initmodel foo
    parser = argparse.ArgumentParser()
    parser.add_argument('val', help='text file')
    parser.add_argument('--initmodel', default='result/model_epoch_500',
                        help='Initialize the model from given file')
    parser.add_argument('--arch', '-a', default='cnn',
                        help='Network architecture')
    args = parser.parse_args()

    # ネットワーク定義はnets.pyを参照
    archs = {
        'mlp': nets.MLP,
        'cnn': nets.ConvNet,
        'cnnbn': nets.ConvNetBN,
    }

    model = archs[args.arch]()

    # 学習済みモデルをロードする
    chainer.serializers.load_npz(args.initmodel, model)
    model.train = False
    model.predict = True

    # Chainerのidxとひらがなの辞書
    labels = {}
    with open('labels.txt', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            c = line.rstrip('\n')
            labels[int(idx)] = c

    # カウント用の変数
    counter_correct = 0
    counter_wrong = 0

    # 推論開始
    with open(args.val, 'r') as f:
        for line in f.readlines():
            imgpath, idx = line.split(' ')
            idx = int(idx.rstrip('\n'))
            label = labels[idx]

            # 画像読込
            image = Image.open(imgpath)
            image = image.convert("L")  # グレースケール
            image = (np.asarray(image, dtype=np.float32) - 127) / 128
            image = image.reshape(32, 32, 1)  # [32,32,1]
            image = image.transpose(2, 0, 1)  # [1,32,32]

            # Chainer形式に変換
            x = chainer.Variable(np.array([image]))

            # 推論
            ret = model(x).data[0]

            # 高確率な上位３候補を出すためソート
            rets = zip(range(0, 71), ret)
            rets = sorted(rets, key=lambda x: x[1], reverse=True)

            # 正解していたらスキップ
            if labels[rets[0][0]] == label:
                counter_correct += 1
            else:
                # 間違えてしまった内容を表示
                counter_wrong += 1
                print('============================')
                print("File  :{}".format(imgpath))
                print("Label :{}".format(label))

                # 上位３を表示
                for i, prob in rets[0:3]:
                    print("{:02d},{:>2} :{:.4f}%".format(i, labels[i], prob))

    print('============================')
    print('accuracy:{:.3f}%({:}/{:})'.format(
        1.0 - (counter_wrong / (counter_correct + counter_wrong)),
        counter_correct, counter_correct + counter_wrong))

if __name__ == '__main__':
    main()
