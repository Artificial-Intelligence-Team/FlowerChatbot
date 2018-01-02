# -*- coding:utf-8 -*-
import numpy as np
import time
import sys
import os
import re
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from dynamic_seq2seq_model import dynamicSeq2seq
import socket
import preprocessing
import io

class Seq2seq():
	'''
	tensorflow-1.0.0

		args:
		encoder_vec_file	encoder向量文件  
		decoder_vec_file	decoder向量文件
		encoder_vocabulary  encoder词典
		decoder_vocabulary  decoder词典
		model_path		  模型目录
		batch_size		  批处理数
		sample_num		  总样本数
		max_batches		 最大迭代次数
		show_epoch		  保存模型步长

	'''
	def __init__(self):
		print("tensorflow version: ", tf.__version__)
		tf.reset_default_graph()
		
		self.encoder_vec_file = "./preprocessing/encode.vector"
		self.decoder_vec_file = "./preprocessing/decode.vector"
		self.encoder_vocabulary = "./preprocessing/encode.vocabulary"
		self.decoder_vocabulary = "./preprocessing/decode.vocabulary"
		self.batch_size = 1
		self.max_batches = 10000
		self.show_epoch = 1000
		self.model_path = './model/'
		self.model = dynamicSeq2seq(encoder_cell=LSTMCell(40),
									decoder_cell=LSTMCell(40), 
									encoder_vocab_size=600,
									decoder_vocab_size=1600,
									embedding_size=20,
									attention=False,
									bidirectional=False,
									debug=False,
									time_major=True)
		self.dec_vocab = {}
		self.enc_vocab = {}
		self.dec_vecToSeg = {}
		tag_location = ''
		with io.open(self.encoder_vocabulary, "r", encoding = "utf-8") as enc_vocab_file:
			for index, word in enumerate(enc_vocab_file.readlines()):
				self.enc_vocab[word.strip()] = index
		with io.open(self.decoder_vocabulary, "r", encoding = "utf-8") as dec_vocab_file:
			for index, word in enumerate(dec_vocab_file.readlines()):
				self.dec_vecToSeg[index] = word.strip()
				self.dec_vocab[word.strip()] = index
		
	def data_set(self, file):
		_ids = []
		with io.open(file, "r") as fw:
			line = fw.readline()
			while line:
				sequence = [int(i) for i in line.split()]
				_ids.append(sequence)
				line = fw.readline()
		return _ids		

	def get_fd(self, train_inputs,train_targets, batches, sample_num):
		'''获取batch

			为向量填充PAD	
			最大长度为每个batch中句子的最大长度  
			并将数据作转换:  
			[batch_size, time_steps] -> [time_steps, batch_size]

		'''
		batch_inputs = []
		batch_targets = []
		batch_inputs_length = []
		batch_targets_length = []

		pad_inputs = []
		pad_targets = []

		# 随机样本
		shuffle = np.random.randint(0, sample_num, batches)
		en_max_seq_length = max([len(train_inputs[i]) for i in shuffle])
		de_max_seq_length = max([len(train_targets[i]) for i in shuffle])

		for index in shuffle:
			_en = train_inputs[index]
			inputs_batch_major = np.zeros(shape=[en_max_seq_length], dtype=np.int32) # == PAD
			for seq in range(len(_en)):
				inputs_batch_major[seq] = _en[seq]
			batch_inputs.append(inputs_batch_major)
			batch_inputs_length.append(len(_en))

			_de = train_targets[index]
			inputs_batch_major = np.zeros(shape=[de_max_seq_length], dtype=np.int32) # == PAD
			for seq in range(len(_de)):
				inputs_batch_major[seq] = _de[seq]
			batch_targets.append(inputs_batch_major)
			batch_targets_length.append(len(_de))
			
		batch_inputs = np.array(batch_inputs).swapaxes(0, 1)
		batch_targets = np.array(batch_targets).swapaxes(0, 1)
		return {self.model.encoder_inputs: batch_inputs,
				self.model.encoder_inputs_length: batch_inputs_length,
				self.model.decoder_targets: batch_targets,
				self.model.decoder_targets_length:batch_targets_length,}

	def train(self):
		# 获取输入输出
		train_inputs = self.data_set(self.encoder_vec_file)
		train_targets = self.data_set(self.decoder_vec_file) 
		
		f = io.open(self.encoder_vec_file, 'r', encoding = "utf-8")
		self.sample_num = len(f.readlines())
		f.close()
		print("共有 %s 樣本" % self.sample_num)


		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True


		with tf.Session(config=config) as sess:
			
			# 初始化变量
			ckpt = tf.train.get_checkpoint_state(self.model_path)
			if ckpt is not None:
				print(ckpt.model_checkpoint_path)
				self.model.saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				sess.run(tf.global_variables_initializer())

			loss_track = []
			total_time = 0
			for batch in range(self.max_batches+1):
				# 获取fd [time_steps, batch_size]
				start = time.time()
				fd = self.get_fd(train_inputs,
								 train_targets,
								 self.batch_size,
								 self.sample_num)
				_, loss,_,_ = sess.run([self.model.train_op, 
										self.model.loss,
										self.model.gradient_norms,
										self.model.updates], fd)
				
				stop = time.time()
				total_time += (stop-start)

				loss_track.append(loss)
				if batch == 0 or batch % self.show_epoch == 0:
					
					print("-" * 50)
					print("n_epoch {}".format(sess.run(self.model.global_step)))
					print('  minibatch loss: {}'.format(sess.run(self.model.loss, fd)))
					print('  per-time: %s'% (total_time/self.show_epoch))
					checkpoint_path = self.model_path + "chatbot_seq2seq.ckpt"
					# 保存模型
					self.model.saver.save(sess, checkpoint_path, global_step=self.model.global_step)

					# 清理模型
					self.clearModel()
					total_time = 0
					for i, (e_in, dt_pred) in enumerate(zip(
						fd[self.model.decoder_targets].T,
						sess.run(self.model.decoder_prediction_train, fd).T
					)):
						print('  sample {}:'.format(i + 1))
						print('	dec targets > {}'.format(e_in))
						print('	dec predict > {}'.format(dt_pred))
						if i >= 0:
							break

	def make_inference_fd(self, inputs_seq):
		sequence_lengths = [len(seq) for seq in inputs_seq]
		max_seq_length = max(sequence_lengths)
		batch_size = len(inputs_seq)
		
		inputs_time_major = []
		# PAD
		for sents in inputs_seq:
			inputs_batch_major = np.zeros(shape=[max_seq_length], dtype=np.int32) # == PAD
			for index in range(len(sents)):
				inputs_batch_major[index] = sents[index]
			inputs_time_major.append(inputs_batch_major)

		inputs_time_major = np.array(inputs_time_major).swapaxes(0, 1)
		return {self.model.encoder_inputs:inputs_time_major, 
				self.model.encoder_inputs_length:sequence_lengths}
	
	def predict(self, question):
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(self.model_path)
			if ckpt is not None:
				print(ckpt.model_checkpoint_path)
				self.model.saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print("没找到模型")
			#inputs_strs = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", inputs_strs)
			segements = []
			segements.extend(question.split(" "))
			#inputs_vec = [enc_vocab.get(i) for i in segements]
			inputs_vec = []
			for i in segements:
				inputs_vec.append(self.enc_vocab.get(i, self.model.UNK))
			fd = self.make_inference_fd([inputs_vec])
			inf_out = sess.run(self.model.decoder_prediction_inference, fd)
			inf_out = [i[0] for i in inf_out]
			outstrs = ''
			for vec in inf_out:
				if vec == self.model.EOS:
					break
				# 針對 Flower Recommendation 目前以一個為主
				outstrs = self.dec_vecToSeg.get(vec, self.model.UNK)
			return outstrs
				
	def clearModel(self, remain = 3):
		try:
			filelists = os.listdir(self.model_path)
			re_batch = re.compile(r"chatbot_seq2seq.ckpt-(\d+).")
			batch = re.findall(re_batch, ",".join(filelists))
			batch = [int(i) for i in set(batch)]
			if remain == 0:
				for file in filelists:
					if "chatbot_seq2seq" in file:
						os.remove(self.model_path+file)
				os.remove(self.model_path+"checkpoint")
				return
			if len(batch) > remain:
				for bat in sorted(batch)[:-(remain)]:
					for file in filelists:
						if str(bat) in file and "chatbot_seq2seq" in file:
							os.remove(self.model_path+file)
		except Exception as e:
			return

if __name__ == '__main__':
	seq = Seq2seq()
	if sys.argv[1]:
		if sys.argv[1] == 'server':
			pp = preprocessing.Preprocessing()
			print("server run.. ")
			soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Create a socket object
			host = "140.116.245.146" # Get local machine name
			port = 1994 # Reserve a port for your service.
			soc.bind((host, port))   # Bind to the port
			soc.listen(5) # Now wait for client connection.
			while True:
				conn, addr = soc.accept() # Establish connection with client.
				print ("Got connection from",addr)
				question = conn.recv(1024)
				relation = pp.term_relationship(question.decode("utf-8"), 5)
				answer = seq.predict(relation if relation != "" else str(question))
				print(answer)
				conn.send((answer + "\n").encode('utf-8'))
		elif sys.argv[1] == 'retrain':
			seq.clearModel(0)
			seq.train()
		elif sys.argv[1] == 'train':
			seq.train()
		elif sys.argv[1] == 'infer':
			print(seq.predict("朋友"))  
			