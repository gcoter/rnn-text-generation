import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn
import random

import constants
from datamanager import DataManager

class BasicConfig(object):
	def __init__(self,vocabulary_size,num_hidden=constants.NUM_HIDDEN,num_features=constants.NUM_FEATURES,learning_rate=constants.LEARNING_RATE):
		self.vocabulary_size = vocabulary_size
		self.num_hidden = num_hidden
		self.num_features = num_features
		self.learning_rate = learning_rate
		
class TrainingConfig(BasicConfig):
	def __init__(self,vocabulary_size,seq_length=constants.SEQ_LENGTH,batch_size=constants.BATCH_SIZE):
		BasicConfig.__init__(self,vocabulary_size)
		self.model_name = "model"
		self.seq_length = seq_length
		self.batch_size = batch_size
		
class GenerationConfig(BasicConfig):
	def __init__(self,vocabulary_size):
		BasicConfig.__init__(self,vocabulary_size)
		self.model_name = "model"
		self.seq_length = 1
		self.batch_size = 1

class Model(object):
	def __init__(self,config):
		self.config = config
		with tf.name_scope(self.config.model_name):
			with tf.name_scope('X_'):
				self.X_ = tf.placeholder(tf.float32, shape=(None,None,self.config.num_features))
				
				""" From https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py """
				# Prepare data shape to match `rnn` function requirements
				# Current data input shape: (batch_size, seq_length, num_features)
				# Required shape: 'seq_length' tensors list of shape (batch_size, num_features)
				
				# Permuting batch_size and seq_length
				self.transposed_X_ = tf.transpose(self.X_, [1, 0, 2])
				# Reshaping to (seq_length*batch_size, num_features)
				self.reshaped_X_ = tf.reshape(self.transposed_X_, [-1, self.config.num_features])
				# Split to get a list of 'seq_length' tensors of shape (batch_size, num_features)
				self.splited_X_ = tf.split(0, self.config.seq_length, self.reshaped_X_)

			with tf.name_scope('Y_'):
				self.Y_ = tf.placeholder(tf.float32, shape=(None,self.config.vocabulary_size))

			with tf.name_scope('keep_prob'):
				self.keep_prob = tf.placeholder(tf.float32)

			with tf.name_scope('Model'):
				# *** LSTM ***
				with tf.name_scope('LSTM'):
					self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_hidden,state_is_tuple=True)
					self.lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * 1,state_is_tuple=True)
					self.initial_state = self.lstm.zero_state(self.config.batch_size, dtype=tf.float32)
					self.lstm_outputs, self.states = rnn.rnn(self.lstm, self.splited_X_, initial_state=self.initial_state, dtype=tf.float32)
					self.lstm_out = self.lstm_outputs[-1]
				
				# *** DROPOUT ***
				with tf.name_scope('Dropout'):
					self.lstm_out_dropout = tf.nn.dropout(self.lstm_out, self.keep_prob)

				# *** OUTPUT LAYER ***
				with tf.name_scope('Output'):
					self.weights_out = tf.get_variable("weights_out",initializer=tf.random_normal(shape=[self.config.num_hidden,self.config.vocabulary_size]))
					self.biaises_out = tf.get_variable("biaises_out",initializer=tf.random_normal(shape=[self.config.vocabulary_size]))

					self.logits_out = tf.matmul(self.lstm_out_dropout,self.weights_out) + self.biaises_out
					self.predicted_Y = tf.nn.softmax(self.logits_out)

			# *** LOSS ***
			with tf.name_scope('Loss'):
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits_out, self.Y_))

			# *** ACCURACY ***
			# with tf.name_scope('Accuracy'):
				# self.accuracy = tf.equal(tf.argmax(self.predicted_Y, 1), tf.argmax(self.Y_, 1))
				# self.mean_accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
				
			# *** TRAIN STEP ***
			with tf.name_scope('Train_step'):
				self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

			# *** SUMMARIES ***
			# tf.scalar_summary("loss", self.loss)
			# tf.scalar_summary("mean_accuracy", self.mean_accuracy)
			# self.merged_summary_op = tf.merge_all_summaries()

			# *** INITIALIZATION ***
			self.init = tf.global_variables_initializer()

"""
A GenerationModel has two sub-models: one model which is batch trainable and one model for generation. 
Both models share the same parameters.
"""
class GenerationModel(object):
	def __init__(self,vocabulary_size,seq_length=constants.SEQ_LENGTH,batch_size=constants.BATCH_SIZE):
		with tf.variable_scope("models") as scope:
			self.training_submodel = Model(TrainingConfig(vocabulary_size,seq_length=seq_length,batch_size=batch_size))
			scope.reuse_variables()
			self.generation_submodel = Model(GenerationConfig(vocabulary_size))

	def sample(self,probabilities,temperature=1.0):
		probabilities = np.log(probabilities+1e-30) / temperature
		probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
		r = random.random() # range: [0,1)
		total = 0.0
		for i in range(len(probabilities)):
			total += probabilities[i]
			if total>r:
				return i
		return len(probabilities)-1
		
	def init(self,session,train=True):
		session.run(self.training_submodel.init)
		session.run(self.generation_submodel.init)
		
	def train_step(self,session,batch_X,batch_Y,keep_prob):
		return session.run(self.training_submodel.train_step, feed_dict={self.training_submodel.X_: batch_X, self.training_submodel.Y_: batch_Y, self.training_submodel.keep_prob: keep_prob})
		
	def get_loss(self,session,batch_X,batch_Y):
		return session.run(self.training_submodel.loss, feed_dict={self.training_submodel.X_: batch_X, self.training_submodel.Y_: batch_Y, self.training_submodel.keep_prob: 1.0})
			
	def generate(self,session,first_token,size,temperature=1.0):
		generated_text = ""
		states = session.run(self.generation_submodel.initial_state)
		input = np.array([[[DataManager.char_to_int(first_token)]]])
		for i in range(size):
			probabilities, states = session.run([self.generation_submodel.predicted_Y,self.generation_submodel.states], feed_dict={self.generation_submodel.X_: input, self.generation_submodel.initial_state: states, self.generation_submodel.keep_prob: 1.0})
			generated_text_index = self.sample(probabilities[0], temperature=temperature)
			generated_character = DataManager.int_to_char(generated_text_index)
			generated_text += generated_character
			input = np.array([[[generated_text_index]]])
		return generated_text