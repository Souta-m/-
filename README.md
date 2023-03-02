# pre-experiments
誤り検出の予備実験用<br>
実行環境などは[松井の修士論文](https://github.com/Souta-m/matsui-master-thesis)と同じもので動きます。主となる部分ではないので、殴り書きになります。
# ファイルとファオルダ説明
順番に簡単な説明をしていきます。
## output
自然言語推論モデルを入れるフォルダです。別途[ダウンロード](https://drive.google.com/drive/folders/1S-hn5aA6fnRFrR3Yy4Cj6Kg6GUHEPtO9?usp=sharing)や構築してください。
## utils
自然言語推論モデルを動かすための機能を格納したファイルです。自然言語推論モデルのコードに関してはほとんど[ここ](https://github.com/yg211/bert_nli)のものと同じです。(バージョンの違い等のエラーにより一部変更)
## bert_nli.py
BERTでの自然言語推論モデルを動作させるための核となるPythonファイルです。
## calc_sihyo.py
F値等を計算するためのテストコードです。
## contra_attention(large).csv,tfjudge.pycontra_attention.csv
矛盾の時のBERTでのattentionの中身を抽出したcsvファイルです。良い結果が得られなかったため、提案システムでは用いていません。
## create_atten.py
attentinoを抽出するためPythonファイルです。
## integ_experiments.py
自然言語推論とSBERTを組み合わせた誤り検出単体での評価をするためのPythonファイルです。
## nli_experiments.py
自然言語推論単体での誤り検出を行うPythonファイルです。
## outsourcing-grammarly.csv
検証データです。
## sbert.py
SBERTが動くかどうかテストするものです。
## sbert_experiments.py
SBERTでの誤り検出を行うPythonファイルです。
## sbert_nli.py
コマンドラインでの入力で閾値による判定するために作成したものです。
## sbert_var.py
文の長さによるSBERTによるcos類似度が変化するかどうか確認するためのPythonファイルです。
## sentence-bert_sim.csv
学習者訳と正解文のcos類似度を示したもので、特に使いません。
## test.py,test_trained_model.py,tfjudge.py,train.py
自然言語推論モデルを訓練したりテストしたりするものです。基本的にモデルはダウンロードすればよいのでtest.pyのみ動かします。
## 以下csv
実験用(本番)のデータです。実験用問題セット.csvを用いればよいです。他の者はcos類似度等を記録したものなので、使わなくて大丈夫です。
