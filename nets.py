import chainer
import chainer.functions as F
import chainer.links as L


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
            l2=L.Linear(None, 71),
        )
        self.train = True
        self.predict = False

    def __call__(self, x, t=None):
        h = self.compute(x)
        if not self.predict:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            return F.softmax(h)

    def compute(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, train=self.train)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, train=self.train)

        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        h = self.l2(h)
        return h


class CNNSample2(chainer.Chain):
    """
    CNN実装
    Conv->Conv->MaxPool->Conv->Conv->MaxPool->FC->FC
    """

    def __init__(self):
        super(CNNSample2, self).__init__(
            conv1=L.Convolution2D(None, 32, 3),
            conv2=L.Convolution2D(None, 32, 3),
            conv3=L.Convolution2D(None, 64, 3),
            conv4=L.Convolution2D(None, 64, 3),
            l1=L.Linear(None, 256, wscale=0.5),
            l2=L.Linear(None, 71, wscale=0.5),  # ここを変更したらいいかな
            # l3=L.Linear(None, 71),
        )
        self.train = True
        self.predict = False

    def __call__(self, x, t=None):
        h = self.compute(x)
        if not self.predict:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            return F.softmax(h)

    def compute(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, train=self.train)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, train=self.train)

        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        h = self.l2(h)
        return h


class CNNSampleBN(chainer.Chain):
    """
    CNN実装
    Conv->Conv->MaxPool->Conv->Conv->MaxPool->FC->FC
    DropoutではなくBatchNormalizationを適用
    """

    def __init__(self):
        super(CNNSampleBN, self).__init__(
            conv1=L.Convolution2D(None, 32, 3),
            conv2=L.Convolution2D(None, 32, 3),
            conv3=L.Convolution2D(None, 64, 3),
            conv4=L.Convolution2D(None, 64, 3),
            l1=L.Linear(None, 256, wscale=0.5),
            l2=L.Linear(None, 71, wscale=0.5),
            bn1=L.BatchNormalization(32),
            bn2=L.BatchNormalization(64),
            # bn3=L.BatchNormalization(256),
        )
        self.train = True
        self.predict = False

    def __call__(self, x, t=None):
        h = self.compute(x)
        if not self.predict:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            return F.softmax(h)

    def compute(self, x):
        test = not self.train

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.bn1(h, test=test)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.bn2(h, test=test)

        h = F.relu(self.l1(h))
        h = F.dropout(h, train=self.train)
        # h = self.bn3(h, test=test)
        h = self.l2(h)
        return h
