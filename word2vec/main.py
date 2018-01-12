# -*- coding: utf-8 -*-
__author__ = "ALEX-CHUN-YU (P76064538@mail.ncku.edu.tw)"
import warnings
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
from gensim.models.keyedvectors import KeyedVectors
import sys, getopt
from train import Train

# 載入 model 並去運用
def main(argv):
	train_text_name = ''
	model_name = ''
	try:
		opts, args = getopt.getopt(argv,"ht:m:",["ifile=","ofile="])
		for opt, arg in opts:
			if opt in ("-t", "--ifile"):
				train_text_name = arg
			if opt in ("-m", "--ofile"):
				model_name = arg
		# 訓練(shallow semantic space)
		if train_text_name != "" and model_name != "":
			t = Train()
			t.train(train_text_name, model_name)
		elif train_text_name == "" and model_name == "":
			sys.exit(2)
		elif train_text_name != "" and model_name == "":
			sys.exit(2)
	except getopt.GetoptError:
		sys.exit(2)
	# 可參考 https://radimrehurek.com/gensim/models/word2vec.html 更多運用
	# How to use bin(model)?
	try:
		word_vectors = KeyedVectors.load_word2vec_format((model_name + ".model.bin"), binary = True)
		while True:
			try:
				query = input("\n輸入格式( Ex: 丁香花,兜蘭,....註:最多三個詞彙)\n")
				query_list = query.split(",")
				if len(query_list) == 1:
					print("此詞彙詞向量為:")
					print(word_vectors[query_list[0]])
					print("詞彙相似詞前 10 排序")
					res = word_vectors.most_similar(query_list[0], topn = 10)
					for item in res:
						print(item[0] + "," + str(item[1]))
				elif len(query_list) == 2:
					print("計算兩個詞彙間 Cosine 相似度")
					res = word_vectors.similarity(query_list[0], query_list[1])
					print(res)
				else:
					print("%s之於%s，如%s之於" % (query_list[0], query_list[1], query_list[2]))
					res = word_vectors.most_similar(positive = [query_list[0], query_list[1]], negative = [query_list[2]], topn = 5)
					for item in res:
						print(item[0] + "," + str(item[1]))
			except Exception as e:
				print("Error:" + repr(e))
	except Exception as e:
		print("沒有此 model 存在!")
	

if __name__ == "__main__":
	print("如要訓練請輸入此格式: test.py -t <train_text_name> -m <model_name>")
	print("如有 model 請輸入此格式: test.py -m <model_name>\n")
	main(sys.argv[1:])