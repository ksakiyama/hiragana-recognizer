import os
import sys
import random
import argparse
import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import nets


class HiraganaDataset(chainer.dataset.DatasetMixin):
    """
    グレースケールの32x32ひらがな画像を読込む
    """

    def __init__(self, path, random=True):
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

    def get_example(self, i):
        crop_size = self.crop_size
        canvas_size = self.canvas_size
        start = self.start
        end = self.end

        imgpath, label = self.base[i]
        image = Image.open(imgpath)
        image = image.convert("L")  # グレースケール
        label = np.int32(label)

        if random:
            # クロッピング
            # 40x40に画像を大きくしてから32x32を抽出する
            canvas = Image.new('L', (canvas_size, canvas_size), 255)
            canvas.paste(image,
                         (int((canvas_size - crop_size) / 2),
                          int((canvas_size - crop_size) / 2)))

            top = random.randint(0, canvas_size - crop_size - 1)
            left = random.randint(0, canvas_size - crop_size - 1)
            bottom = top + crop_size
            right = left + crop_size

            image = canvas.crop((left, top, right, bottom))

        # [-1.0, 1.0]の範囲に値を変換する
        image = (np.asarray(image, dtype=np.float32) - 127) / 128  # numpy形式
        image = image.reshape(crop_size, crop_size, 1)     # [32,32,1]
        image = image.transpose(2, 0, 1)                   # [1,32,32]
        return image, label


class TestModeEvaluator(extensions.Evaluator):
    """
    モデルを評価するためのラッパー。
    """

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    # ここで実行時の引数を設定できます
    # 例：python trainer.py --arch cnn --epoch 10 --gpu 0 [学習データ] [テストデータ]
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=1024,
                        help='Learning minibatch size')
    parser.add_argument('--val_batchsize', '-b', type=int, default=1024,
                        help='Validation minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help='CPU mode')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--arch', '-a', default='mlp',
                        help='Network architecture')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    args = parser.parse_args()

    # ネットワーク定義はnets.pyを参照
    archs = {
        'mlp': nets.MLP,
        'cnn': nets.ConvNet,
        'cnnbn': nets.ConvNetBN,
    }

    model = archs[args.arch]()

    # 学習済みモデルがあればそれをロード
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)

    if args.cpu:
        args.gpu = -1
    else:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # TODO
    # いろいろな最適化アルゴリズムを試してみましょう
    optimizer = chainer.optimizers.Adam()
    # optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    # optimizer = chainer.optimizers.SMORMS3()
    # optimizer = chainer.optimizers.AdaDelta()
    optimizer.setup(model)

    # オリジナルのデータセットクラスを使用
    # TODO クロッピングを有効にしてみましょう
    train_data = HiraganaDataset(args.train, False)
    test_data = HiraganaDataset(args.val, False)

    # イテレータ
    # train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    # test_iter = chainer.iterators.SerialIterator(
    #     test_data, args.val_batchsize, repeat=False, shuffle=False)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, args.batchsize, n_processes=4)
    test_iter = chainer.iterators.MultiprocessIterator(
        test_data, args.val_batchsize, repeat=False, shuffle=False,
        n_processes=4)

    # trainerを定義
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # テスト用のイテレータを登録
    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu),
                   trigger=(1, 'epoch'))

    trainer.extend(extensions.dump_graph('main/loss'))

    # 毎epochでオブジェクトを保存
    every_epoch = (1, 'epoch')
    trainer.extend(extensions.snapshot(
        filename='trainer_{.updater.epoch}'), trigger=every_epoch)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'), trigger=every_epoch)
    trainer.extend(extensions.snapshot_object(
        optimizer, 'optimizer_epoch_{.updater.iteration}'),
        trigger=every_epoch)

    # 毎epochでログ出力
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # 学習状況をかっこよく表示
    trainer.extend(extensions.ProgressBar())

    # 学習の様子をグラフとして保存
    trainer.extend(extensions.dump_graph(
        root_name="main/loss", out_name="cg.dot"))

    # 学習を途中から再開する場合に使用
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # 学習開始
    trainer.run()

if __name__ == '__main__':
    main()
