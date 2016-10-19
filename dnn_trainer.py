# -*- coding: utf-8 -*-
import os
import sys
import random
import argparse
import numpy as np
import cv2

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


class MLP(chainer.Chain):
    """
    Chainerのサンプルから拝借
    https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py
    """

    def __init__(self):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, 1024),  # n_in -> n_units
            l2=L.Linear(None, 1024),  # n_units -> n_units
            l3=L.Linear(None, 75),  # n_units -> n_out
        )

    def __call__(self, x, t):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class CNNSample(chainer.Chain):
    """
    CNN実装
    Conv->Conv->MaxPool->Conv->Conv->MaxPool->FC->FC
    """

    def __init__(self):
        super(CNNSample, self).__init__(
            conv1=L.Convolution2D(None, 32, 3),
            conv2=L.Convolution2D(None, 32, 3),
            conv3=L.Convolution2D(None, 64, 3),
            conv4=L.Convolution2D(None, 64, 3),
            l1=L.Linear(None, 256),
            l2=L.Linear(None, 75),
        )
        self.train = True

    def __call__(self, x, t):
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
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class HiraganaDataset(chainer.dataset.DatasetMixin):
    """
    グレースケールの32x32ひらがな画像を読込む
    """

    def __init__(self, path, root, random=True):
        canvas_size = 40
        crop_size = 32

        base = []
        with open(path, 'r') as f:
            for line in f.readlines():
                imgpath, label = line.split(' ')
                label = int(label.rstrip('\n'))
                base.append((imgpath, label))

        self.base = base

        self.crop_size = crop_size
        self.canvas_size = canvas_size
        self.random = random
        self.canvas = np.ones((canvas_size, canvas_size, 1), dtype=np.float32)

        self.start = int((canvas_size - crop_size) / 2)
        self.end = canvas_size - self.start

    def __len__(self):
        return len(self.base)

    def _read_gray_image_as_array(self, imgpath):
        crop_size = self.crop_size
        cvimg = cv2.imread(imgpath, 0)
        cvimg = cv2.resize(cvimg, (crop_size, crop_size))
        cvimg = cvimg.astype(np.float32)
        cvimg = cvimg / 255
        return cvimg.reshape(crop_size, crop_size, 1)

    def get_example(self, i):
        crop_size = self.crop_size
        canvas_size = self.canvas_size
        start = self.start
        end = self.end

        imgpath, label = self.base[i]
        image = self._read_gray_image_as_array(imgpath)
        label = np.int32(label)

        # -10〜10度で回転させ、そこからクロッピングする
        if self.random:
            # 40x40のキャンバスの中央に文字貼り付け
            canvas = np.copy(self.canvas)
            canvas[start:end, start:end, :] = image
            image = canvas

            # いったん白黒を反転（warpAffineの背景が黒のため仕方なく。。）
            image = 1 - image

            # 乱数で回転を実施
            rotate = random.randint(-10, 10)
            rotation_matrix = cv2.getRotationMatrix2D(
                (canvas_size / 2, canvas_size / 2), rotate, 1)
            image = cv2.warpAffine(
                image, rotation_matrix, (canvas_size, canvas_size))

            # 白黒を元に戻す
            image = 1 - image

            # 32x32を抜き出す
            dif = canvas_size - crop_size - 2
            top = random.randint(0, dif)
            left = random.randint(0, dif)
            bottom = top + crop_size
            right = left + crop_size
            image = image[top:bottom, left:right]
            image = image.reshape(crop_size, crop_size, 1)

        return image.transpose(2, 0, 1), label


def main():
    parser = argparse.ArgumentParser(description='ひらがなの学習')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--batchsize', '-B', type=int, default=2048,
                        help='Learning minibatch size')
    parser.add_argument('--val_batchsize', '-b', type=int, default=1024,
                        help='Validation minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--arch', '-a', default='cnn',
                        help='Network architecture')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    args = parser.parse_args()

    archs = {
        'cnn': CNNSample,
        'mlp': MLP
    }

    # モデルの初期化
    model = archs[args.arch]()

    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)

    # GPUを使う場合
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # オリジナルのデータセットクラスを使用
    train_data = HiraganaDataset(args.train, args.root)
    test_data = HiraganaDataset(args.val, args.root, False)

    # イテレータ
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test_data, args.val_batchsize, repeat=False, shuffle=False)

    # trainerを定義
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # テスト用のイテレータを登録
    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu),
                   trigger=(1, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # 定期的にオブジェクトを保存する
    trainer.extend(extensions.snapshot(
        filename='trainer_{.updater.epoch}'), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        optimizer, 'optimizer_epoch_{.updater.iteration}'), trigger=(10, 'epoch'))

    # 毎epochでログ出力
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
