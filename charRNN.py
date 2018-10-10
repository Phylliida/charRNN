# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import uuid
import time
import math
import numpy as np
import my_txtutils as txt
import json
# model parameters
#
# Usage:
#   Training only:
#         Leave all the parameters as they are
#         Disable validation to run a bit faster (set validation=False below)
#         You can follow progress in Tensorboard: tensorboard --log-dir=log
#   Training and experimentation (default):
#         Keep validation enabled
#         You can now play with the parameters anf follow the effects in Tensorboard
#         A good choice of parameters ensures that the testing and validation curves stay close
#         To see the curves drift apart ("overfitting") try to use an insufficient amount of
#         training data (shakedir = "shakespeare/t*.txt" for example)
#
'''
seqLen = 30
batchSize = 200
alphaSize = txt.alphaSize
internalSize = 512
nLayers = 3
learningRate = 0.001  # fixed learning rate
dropoutPkeep = 0.8    # some dropout
'''



class TextData(object):
	def __init__(self, rootDir, batchSize, seqLen):
		self.batchSize = batchSize
		self.seqLen = seqLen
		self.rootDir = rootDir
		# load data, either shakespeare, or the Python source of Tensorflow itself
		self.textdir = self.rootDir + "/*.txt"
		#shakedir = "../tensorflow/**/*.py"
		self.codetext, self.valitext, self.bookranges = txt.read_data_files(self.textdir, validation=True)
		self.epoch_size = len(self.codetext) // (self.batchSize * self.seqLen)

		# display some stats on the data
		txt.print_data_stats(len(self.codetext), len(self.valitext), self.epoch_size)

# holds all of the tensorflow variables
class RNNTFStuff(object):
	def __init__(self):
		pass
#
# the model (see FAQ in README.md)
#
class CharRNN(object):
	
	def loadData(self, rootDir):
		return TextData(rootDir, self.batchSize, self.seqLen)
	
	def __init__(self, name, seqLen=30, batchSize=200, alphaSize=txt.ALPHASIZE, internalSize=512, nLayers=3, learningRate=0.001, dropoutPkeep=0.8, randomSeed=0, prevPath=None):
		self.seqLen = seqLen
		self.batchSize = batchSize
		self.alphaSize = alphaSize
		self.internalSize = internalSize
		self.nLayers = nLayers
		self.learningRate = learningRate
		self.dropoutPkeep = dropoutPkeep
		self.name = name
		self.randomSeed = randomSeed
		# if we are loading a prevous model, fetch the params it used
		if not prevPath is None:
			if os.path.isdir(prevPath):
				pass
			f = open(prevPath + ".json", "r")
			self.data = json.load(f)
			f.close()
			for k,v in self.data.items():
				print("restoring " + str(k) + " to " + str(v))
				setattr(self, k, v)
		
		else:
			self.data = {"seqLen": seqLen, "batchSize": batchSize, "alphaSize": alphaSize, "internalSize": internalSize, "nLayers": nLayers, "learningRate": learningRate, "dropoutPkeep": dropoutPkeep, "randomSeed": randomSeed}
		self.tfStuff = tfStuff = RNNTFStuff()
		self.tbStuff = TensorboardStuff(self.name)
		# Init for saving models. They will be saved into a directory named 'checkpoints'.
		# Only the last checkpoint is kept.
		# a scope allows us to have multiple rnns im memory at the same time
		#self.scopeName = str(uuid.uuid4())
		#self.fullName = self.name + "-" + self.scopeName
		self.fullName = "rnnVars"
		# making a seperate graph for each instance of this class makes saving and loading nice
		self.tfStuff.graph = tf.Graph()
		self.checkpointsPath = "checkpoints/" + self.name
		with self.tfStuff.graph.as_default():
			#with tf.name_scope(self.scopeName):
			with tf.variable_scope(self.fullName, reuse=tf.AUTO_REUSE):
				tf.set_random_seed(self.randomSeed)
				tfStuff.lr = lr = tf.placeholder(tf.float32, name='lr')  # learning rate
				tfStuff.pkeep = pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
				tfStuff.batchsize = batchsize = tf.placeholder(tf.int32, name='batchsize')

				# inputs
				tfStuff.X = X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ batchSize, seqLen ]
				tfStuff.Xo = Xo = tf.one_hot(X, self.alphaSize, 1.0, 0.0)                 # [ batchSize, seqLen, alphaSize ]
				# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
				tfStuff.Y_ = Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ batchSize, seqLen ]
				tfStuff.Yo_ = Yo_ = tf.one_hot(Y_, self.alphaSize, 1.0, 0.0)               # [ batchSize, seqLen, alphaSize ]
				# input state
				tfStuff.Hin = Hin = tf.placeholder(tf.float32, [None, self.internalSize * self.nLayers], name='Hin')  # [ batchSize, internalSize * nLayers]

				# using a nLayers=3 layers of GRU cells, unrolled seqLen=30 times
				# dynamic_rnn infers seqLen from the size of the inputs Xo

				# How to properly apply dropout in RNNs: see README.md
				tfStuff.cells = cells = [rnn.GRUCell(self.internalSize) for _ in range(self.nLayers)]
				# "naive dropout" implementation
				tfStuff.dropcells = dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
				multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
				tfStuff.multicell = multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

				Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
				tfStuff.Yr = Yr
				# Yr: [ batchSize, seqLen, internalSize ]
				# H:  [ batchSize, internalSize*nLayers ] # this is the last state in the sequence

				tfStuff.H = H = tf.identity(H, name='H')  # just to give it a name

				# Softmax layer implementation:
				# Flatten the first two dimension of the output [ batchSize, seqLen, alphaSize ] => [ batchSize x seqLen, alphaSize ]
				# then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
				# From the readout point of view, a value coming from a sequence time step or a minibatch item is the same thing.

				tfStuff.Yflat = Yflat = tf.reshape(Yr, [-1, self.internalSize])    # [ batchSize x seqLen, internalSize ]
				tfStuff.Ylogits = Ylogits = layers.linear(Yflat, self.alphaSize)     # [ batchSize x seqLen, alphaSize ]
				tfStuff.Yflat_ = Yflat_ = tf.reshape(Yo_, [-1, self.alphaSize])     # [ batchSize x seqLen, alphaSize ]
				loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ batchSize x seqLen ]
				tfStuff.loss = loss = tf.reshape(loss, [batchsize, -1])      # [ batchSize, seqLen ]
				tfStuff.Yo = Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ batchSize x seqLen, alphaSize ]
				Y = tf.argmax(Yo, 1)                          # [ batchSize x seqLen ]
				tfStuff.Y = Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ batchSize, seqLen ]
				tfStuff.train_step = train_step = tf.train.AdamOptimizer(lr).minimize(loss)

				# stats for display
				tfStuff.seqloss = tf.reduce_mean(tfStuff.loss, 1)
				tfStuff.batchloss = tf.reduce_mean(tfStuff.seqloss)
				tfStuff.accuracy = tf.reduce_mean(tf.cast(tf.equal(tfStuff.Y_, tf.cast(tfStuff.Y, tf.uint8)), tf.float32))
				tfStuff.loss_summary = tf.summary.scalar("batch_loss", tfStuff.batchloss)
				tfStuff.acc_summary = tf.summary.scalar("batch_accuracy", tfStuff.accuracy)
				tfStuff.summaries = tf.summary.merge([tfStuff.loss_summary, tfStuff.acc_summary])


				# init (this maybe should be in __init__ instead of here? I'm not not sure how sessions work yet)
				tfStuff.istate = np.zeros([self.batchSize, self.internalSize * self.nLayers])  # initial zero input state

				init = tf.global_variables_initializer()
				tfStuff.sess = tf.Session(graph=tfStuff.graph)
				tfStuff.sess.run(init)
				# keep 5 most recent checkpoints only
				tfStuff.saver = tf.train.Saver(max_to_keep=5)
				
				# load a previous model
				if not prevPath is None:
					tfStuff.saver.restore(tfStuff.sess, prevPath)
				else:
					self.step = 0
					self.curBatch = 0
					self.curEpoch = 0

			

	def save(self, checkpointsPath=None, alreadyInGraph=False):
		tfStuff = self.tfStuff
		if alreadyInGraph:
			if checkpointsPath is None:
				if not os.path.exists("checkpoints"):
					os.mkdir("checkpoints")
				if not os.path.exists(self.checkpointsPath):
					os.mkdir(self.checkpointsPath)
				path = self.checkpointsPath + '/rnn_train_' + self.name + "-" + str(self.step)
			else:
				if not os.path.exists(checkpointsPath):
					os.mkdir(checkpointsPath)
				path = checkpointsPath + '/rnn_train_' + self.name + "-" + str(self.step)
			saved_file = tfStuff.saver.save(tfStuff.sess, path)
			self.data['step'] = self.step
			self.data['curBatch'] = self.curBatch
			self.data['curEpoch'] = self.curEpoch
			f = open(path + ".json", "w")
			json.dump(self.data, f)
			f.close()
			print("Saved file: " + path)

		else:
			with self.tfStuff.graph.as_default():
				with tf.variable_scope(self.fullName, reuse=tf.AUTO_REUSE):
					self.save(checkpointsPath=checkpointsPath, alreadyInGraph=True)

	
	def generate(self, len, seed="K", topn=2):
		with self.tfStuff.graph.as_default():
			with tf.variable_scope(self.fullName, reuse=tf.AUTO_REUSE):
				tfStuff = self.tfStuff
				sess = tfStuff.sess
				ry = np.array([[txt.convert_from_alphabet(ord(seed[0]))]])
				rh = np.zeros([1, self.internalSize * self.nLayers])
				res = []
				for k in range(len):
					ryo, rh = sess.run([tfStuff.Yo, tfStuff.H], feed_dict={tfStuff.X: ry, tfStuff.pkeep: 1.0, tfStuff.Hin: rh, tfStuff.batchsize: 1})
					rc = txt.sample_from_probabilities(ryo, topn=topn)
					cur = chr(txt.convert_to_alphabet(rc))
					res.append(cur)
					print(cur, end="")
					ry = np.array([[rc]])
				return "".join(res)
	
	def fit(self, data, epochs=1000, displayFreq=50, genFreq=150, saveFreq=5000, verbosity=2):
		progress = txt.Progress(displayFreq, size=111+2, msg="Training on next "+str(displayFreq)+" batches")
		tfStuff = self.tfStuff
		valitext = data.valitext
		# todo: check if batchSize != data.batchSize or if seqLen != data.seqLen (if so, I think we need to raise an exception?)
		firstEpoch = self.curEpoch
		lastEpoch = firstEpoch + epochs
		isFirstStepInThisFitCall = True
		try:
			with tfStuff.graph.as_default():
				#with tf.name_scope(self.scopeName):
				with tf.variable_scope(self.fullName, reuse=tf.AUTO_REUSE):
					
					sess = tfStuff.sess
					
					# training loop
					for x, y_, epoch, batch in txt.rnn_minibatch_sequencer(data.codetext, self.batchSize, self.seqLen, nb_epochs=epochs, startBatch=self.curBatch, startEpoch=self.curEpoch):
						
						nSteps = self.step // (self.batchSize*self.seqLen)
						# train on one minibatch
						feed_dict = {tfStuff.X: x, tfStuff.Y_: y_, tfStuff.Hin: tfStuff.istate, tfStuff.lr: self.learningRate, tfStuff.pkeep: self.dropoutPkeep, tfStuff.batchsize: self.batchSize}
						_, y, ostate = sess.run([tfStuff.train_step, tfStuff.Y, tfStuff.H], feed_dict=feed_dict)

						# log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
						if nSteps % displayFreq == 0 or isFirstStepInThisFitCall:
							feed_dict = {tfStuff.X: x, tfStuff.Y_: y_, tfStuff.Hin: tfStuff.istate, tfStuff.pkeep: 1.0, tfStuff.batchsize: self.batchSize}  # no dropout for validation
							y, l, bl, acc, smm = sess.run([tfStuff.Y, tfStuff.seqloss, tfStuff.batchloss, tfStuff.accuracy, tfStuff.summaries], feed_dict=feed_dict)
							txt.print_learning_learned_comparison(x, y, l, data.bookranges, bl, acc, data.epoch_size, self.step, epoch, lastEpoch, verbosity=verbosity)
							self.tbStuff.summary_writer.add_summary(smm, self.step)
						# run a validation step every 50 batches
						# The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
						# so we cut it up and batch the pieces (slightly inaccurate)
						# tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
						
						if (nSteps % displayFreq == 0 or isFirstStepInThisFitCall) and len(data.valitext) > 0:
							VALI_seqLen = 1*1024  # Sequence length for validation. State will be wrong at the start of each sequence.
							bsize = len(data.valitext) // VALI_seqLen
							if verbosity >= 1: txt.print_validation_header(len(data.codetext), data.bookranges)
							vali_x, vali_y, _, _ = next(txt.rnn_minibatch_sequencer(data.valitext, bsize, VALI_seqLen, 1))  # all data in 1 batch
							vali_nullstate = np.zeros([bsize, self.internalSize * self.nLayers])
							feed_dict = {tfStuff.X: vali_x, tfStuff.Y_: vali_y, tfStuff.Hin: vali_nullstate, tfStuff.pkeep: 1.0,  # no dropout for validation
										 tfStuff.batchsize: bsize}
							ls, acc, smm = sess.run([tfStuff.batchloss, tfStuff.accuracy, tfStuff.summaries], feed_dict=feed_dict)
							if verbosity >= 1: txt.print_validation_stats(ls, acc)
							# save validation data for Tensorboard
							self.tbStuff.validation_writer.add_summary(smm, self.step)

						# display a short text generated with the current weights and biases (every 150 batches)
						if nSteps % genFreq == 0 or isFirstStepInThisFitCall:
							txt.print_text_generation_header()
							ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
							rh = np.zeros([1, self.internalSize * self.nLayers])
							for k in range(1000):
								ryo, rh = sess.run([tfStuff.Yo, tfStuff.H], feed_dict={tfStuff.X: ry, tfStuff.pkeep: 1.0, tfStuff.Hin: rh, tfStuff.batchsize: 1})
								rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
								print(chr(txt.convert_to_alphabet(rc)), end="")
								ry = np.array([[rc]])
							txt.print_text_generation_footer()
						if isFirstStepInThisFitCall:
							for i in range(nSteps % displayFreq):
								progress.step()
							isFirstStepInThisFitCall = False
						# save a checkpoint (every 500 batches)
						if nSteps % saveFreq == 0:
							self.save(alreadyInGraph=True)

						# display progress bar
						progress.step(reset=nSteps % displayFreq == 0)

						# loop state around
						tfStuff.istate = ostate
						self.step += self.batchSize * self.seqLen
						self.curEpoch = epoch
						self.curBatch = batch
		except KeyboardInterrupt as e:
			print("\npressed ctrl-c, saving")
			self.save()


class TensorboardStuff(object):
	def __init__(self, name):
		# Init Tensorboard stuff. This will save Tensorboard information into a different
		# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
		# you can compare training and validation curves visually in Tensorboard.
		if not os.path.exists("log"):
			os.mkdir("log")
		self.label = name
		self.summary_writer = tf.summary.FileWriter("log/" + self.label + "-training")
		self.validation_writer = tf.summary.FileWriter("log/" + self.label + "-validation")
		
		
		
		# for display: init the progress bar
		#self.displayFreq = 50
		#self.batchSize = batchSize
		#self.seqLen = seqLen
		#self.batches50 = self.displayFreq * self.batchSize * self.seqLen
# all runs: seqLen = 30, batchSize = 100, alphaSize = 98, internalSize = 512, nLayers = 3
# run 1477669632 decaying learning rate 0.001-0.0001-1e7 dropout 0.5: not good
# run 1477670023 lr=0.001 no dropout: very good

# Tensorflow runs:
# 1485434262
#   trained on shakespeare/t*.txt only. Validation on 1K sequences
#   validation loss goes up from step 5M (overfitting because of small dataset)
# 1485436038
#   trained on shakespeare/t*.txt only. Validation on 5K sequences
#   On 5K sequences validation accuracy is slightly higher and loss slightly lower
#   => sequence breaks do introduce inaccuracies but the effect is small
# 1485437956
#   Trained on shakespeare/*.txt. Validation on 1K sequences
#   On this much larger dataset, validation loss still decreasing after 6 epochs (step 35M)
# 1495447371
#   Trained on shakespeare/*.txt no dropout, 30 epochs
#   Validation loss starts going up after 10 epochs (overfitting)
# 1495440473
#   Trained on shakespeare/*.txt "naive dropout" pkeep=0.8, 30 epochs
#   Dropout brings the validation loss under control, preventing it from
#   going up but the effect is small.


