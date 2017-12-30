# 基於 flower chatbot 所架設出來的 tensorflow model 伺服器
主要根據較為模糊的情境所進行的運用。EX: 我要去參加婚禮要送甚麼花呢?

## 使用方式
```
# 預先處理
python preprocessing.py
# 啟動伺服器
python seq2seq.py server
# 訓練
python seq2seq.py train
# 重新訓練
python seq2seq.py retrain
# 預測
python seq2seq.py infer
```
	
## 使用 chiachun1127 作者 dynamic-seq2seq structure 進行改寫
https://github.com/chiachun1127/dynamic-seq2seq
