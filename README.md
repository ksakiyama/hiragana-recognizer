# hiragana-recognizer
ひらがなを学習するCNNです。Chainerを使って実装しております。

### ファイルの内容
* dnn_trainer.py
  * DNNを学習させるスクリプト
* recognizer.py
  + dnn_trainer.pyで学習させたネットワークを使い、間違えた画像とその予測値を表示する
* separate_data.py
  + imgs/ディレクトリを元にtraindata.txtとtestdata.txtを作成する

### 動かした環境
* Ubuntu 14.04LTS
* Python 3.5.2 (Anaconda)
* Chainer 1.17.0

### 参考にしたURL
[手書きひらがなの認識で99.78%の精度をディープラーニングで](http://qiita.com/yukoba/items/7a687e44395783eb32b1)

### データ準備
* [ETLデータセット](http://etlcdb.db.aist.go.jp/?page_id=2461)  
* 上記からダウンロードしてください
* サイトに画像復元のPythonスクリプトの記載があります
* 各画像をimgsディレクトリを作って保存してください
  + ファイル名は「ETL8B2_${ラベル番号}_${連番}.png」としてください
  + 例：「ETL8B2_00_000.png」

### 使い方
例えば、以下の引数で実行するとGPU0番で10epochの学習を実施します
```
python dnn_trainer.py traindata.txt testdata.txt --gpu 0 --epoch 10
```
