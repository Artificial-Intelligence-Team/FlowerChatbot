# -*- coding:utf-8 -*-
import re
import warnings
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
from gensim.models.keyedvectors import KeyedVectors
import io

class Preprocessing():
	__PAD__ = 0
	__GO__ = 1
	__EOS__ = 2
	__UNK__ = 3
	vocab = ['__PAD__', '__GO__', '__EOS__', '__UNK__']
	def __init__(self):
		self.encoderFile = "./question.txt"
		self.decoderFile = './answer.txt'
		self.stopwordsFile = "./preprocessing/stopwords.dat"

	def term_relationship(self, term_list, rank_number):
		print(term_list)
		word_vectors = KeyedVectors.load_word2vec_format(("flower" + ".model.bin"), binary = True)
		relation = ""
		for term in term_list.split(" "):
			try:
				res = word_vectors.most_similar(term, topn = rank_number)
				for item in res:
					relation += (item[0] + " ")
			except Exception as e:
				print(e)
		return relation

	def read_wordtovec_model(self, decodeFile, encodeFile):
		word_vectors = KeyedVectors.load_word2vec_format(("flower" + ".model.bin"), binary = True)
		question = io.open(encodeFile, "w", encoding = "utf-8")
		with io.open(decodeFile, 'r', encoding = "utf-8") as decode:
			for answer in decode.readlines():
				print(answer)
				for a in answer.split(' '):
					if a.strip() == "":
						break;
					try:
						res = word_vectors.most_similar(a.strip(), topn = 10)
					except Exception as e:
						print(e)
					for item in res:
						print(item[0] + "," + str(item[1]))
						question.write(item[0] + " ")
				question.write("\n")
				print("-----------------------------------------")
		'''
		while True:
			try:
				query = input("\n輸入格式( Ex: 丁香花,兜蘭,....註:最多三個詞彙)\n")
				query_list = query.split(",")
				if len(query_list) == 1:
					print("此詞彙詞向量為:")
					print(word_vectors[query_list[0]])
					print("詞彙相似詞前 20 排序")
					res = word_vectors.most_similar(query_list[0], topn = 20)
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
		'''

	# original File 裡暫且不用到
	def word_to_vocabulary(self, originFile, vocabFile, segementFile):
		vocabulary = []
		# stopwords = [i.strip() for i in open(self.stopwordsFile).readlines()]
		# print(stopwords)
		# exit()
		'''
		sege = open(segementFile, "w", encoding = "utf-8")
		with open(originFile, 'r', encoding = "utf-8") as en:
			for sent in en.readlines():
				# 去标点
				if "enc" in segementFile:
					#sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent.strip())
					sentence = sent.strip()
					words = jieba.lcut(sentence)
					print(words)
				else:
					words = jieba.lcut(sent.strip())
					print(words)
				vocabulary.extend(words)
				for word in words:
					sege.write(word+" ")
				sege.write("\n")
		sege.close()
		'''
		# Segement Dic 裡要先斷好存進去 
		with io.open(segementFile, 'r', encoding = "utf-8") as segment:
			for seg in segment.readlines():
				for s in seg.split(' '):
					vocabulary.append(s.strip())
		# 產生 Bag 並寫入檔案中
		vocab_file = io.open(vocabFile, "w", encoding = "utf-8")
		_vocabulary = list(set(vocabulary))
		_vocabulary.sort(key = vocabulary.index)
		_vocabulary = self.vocab + _vocabulary
		for index, word in enumerate(_vocabulary):
			vocab_file.write(word + "\n")
		vocab_file.close()

	def to_vec(self, segementFile, vocabFile, doneFile):
		word_dicts = {}
		vec = []
		with io.open(vocabFile, "r", encoding = "utf-8") as dict_f:
			for index, word in enumerate(dict_f.readlines()):
				word_dicts[word.strip()] = index

		f = io.open(doneFile, "w", encoding = "utf-8")
		with io.open(segementFile, "r", encoding = "utf-8") as sege_f:
			for sent in sege_f.readlines():
				sents = [i.strip() for i in sent.split(" ")[:-1]]
				vec.extend(sents)
				for word in sents:
					f.write(str(word_dicts.get(word))+" ")
				f.write("\n")
		f.close()

	def main(self):
		# 透過 decode segement 丟入 word2vec model 產生 tensorflow question
		self.read_wordtovec_model('./preprocessing/decode.segement', './preprocessing/encode.segement')
		# 获得字典
		self.word_to_vocabulary(self.encoderFile, './preprocessing/encode.vocabulary', './preprocessing/encode.segement')
		self.word_to_vocabulary(self.decoderFile, './preprocessing/decode.vocabulary', './preprocessing/decode.segement')
		# Sentence To Vec
		self.to_vec("./preprocessing/encode.segement", 
				   "./preprocessing/encode.vocabulary", 
				   "./preprocessing/encode.vector")
		self.to_vec("./preprocessing/decode.segement", 
				   "./preprocessing/decode.vocabulary", 
				   "./preprocessing/decode.vector")

if __name__ == '__main__':
	pre = Preprocessing()
	pre.main()