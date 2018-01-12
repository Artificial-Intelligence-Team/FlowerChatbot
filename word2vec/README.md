# FlowerChatbot WORD2VEC

## 訓練資料格式
* 花語 + 對象 + 花 + 花語 + 對象

## 目前訓練資料
* 資料連結(http://tools.2345.com/yule/flower.htm)
* 169 朵花經過人工整理成(訓練資料格式) flower.txt
<<<<<<< HEAD
* flower_train_data 經過重覆複製來增進訓練資料可靠信
* 使用 CBOW Alogorithm 方式訓練(如使用 skip gram 效果較差，因格式關西('對象'會未出現))
=======
* flower_train_data 經過重複複製來增進訓練資料可靠信
* 使用 CBOW Alogorithm 方式訓練(如使用 skip gram 效果較差，因格式關係('對象'會未出現))

## 使用方式
```

如要訓練請輸入此格式: test.py -t <train_text_name> -m <output_model_name>
EX: python main.py -t flower_train_data -m flower

如有 model 請輸入此格式: test.py -m <input_model_name>
EX: python main.py -m flower

```
