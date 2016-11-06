# hiragana-recognizer
ひらがなを学習するCNNです。Chainerを使って実装しております。

## ファイルの内容
* dnn_trainer.py
  * DNN(DeepNeuralNetwork)、CNNを学習させるスクリプト
* recognizer.py
  + dnn_trainer.pyで学習させたネットワークを使い、間違えた画像とその予測値を表示する
* text_generator.py
  + imgs/ディレクトリを元にtraindata.txtとtestdata.txtを作成する
* image_restorer.py
  + ETLデータセットから64x64のPNG画像を取得してimgsディレクトリに保存する

## 動かした環境
* Ubuntu 14.04LTS
* Python 3.5.2 (Anaconda)
* Chainer 1.17.0

## 参考にしたURL
[手書きひらがなの認識で99.78%の精度をディープラーニングで](http://qiita.com/yukoba/items/7a687e44395783eb32b1)

## データ準備
* [ETLデータセット](http://etlcdb.db.aist.go.jp/?page_id=2461)  
* 上記からダウンロードしてください
* zipを解答して、image_restorer.pyを実行するとimgsに画像が保存されます
* 任意：text_generator.pyを実行すると、新しいtraindata.txtとtestdata.txtが作成されます）

## 使い方
例えば、以下の引数で実行するとGPU0番で10epochの学習を実施します
```
python dnn_trainer.py traindata.txt testdata.txt --gpu 0 --epoch 10
```

## メモ
* 正解率は最高で99.3%くらいです
* Geforce GTX 980Tiなら10分程度で学習が完了します