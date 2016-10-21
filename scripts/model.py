import tensorflow as tf
from tensorflow.python.ops import rnn

class BasicConfig(object):
	def __init__(self,vocabulary_size):
		self.vocabulary_size = vocabulary_size
		self.num_hidden = 256
		self.num_features = 1
		self.learning_rate = 1e-3
		
class TrainingConfig(BasicConfig):
	def __init__(self,vocabulary_size):
		BasicConfig.__init__(self,vocabulary_size)
		self.model_name = "training_model"
		self.seq_length = 100
		self.batch_size = 128
		
class GenerationConfig(BasicConfig):
	def __init__(self,vocabulary_size):
		BasicConfig.__init__(self,vocabulary_size)
		self.model_name = "generation_model"
		self.seq_length = 1
		self.batch_size = 1

class Model(object):
	def __init__(self,config):
		with tf.name_scope(config.model_name):
			with tf.name_scope('X_'):
				self.X_ = tf.placeholder(tf.float32, shape=(config.batch_size,config.seq_length,config.num_features))
				
				""" From https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py """
				# Prepare data shape to match `rnn` function requirements
				# Current data input shape: (batch_size, n_steps, n_input)
				# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
				
				# Permuting batch_size and n_steps
				self.transposed_X_ = tf.transpose(self.X_, [1, 0, 2])
				# Reshaping to (n_steps*batch_size, n_input)
				self.reshaped_X_ = tf.reshape(self.transposed_X_, [-1, config.num_features])
				# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
				self.splited_X_ = tf.split(0, config.seq_length, self.reshaped_X_)

			with tf.name_scope('Y_'):
				self.Y_ = tf.placeholder(tf.float32, shape=(config.batch_size,config.vocabulary_size))

			with tf.name_scope('keep_prob'):
				self.keep_prob = tf.placeholder(tf.float32)

			with tf.name_scope('Model'):
				# *** LSTM ***
				with tf.name_scope('LSTM'):
					self.lstm = tf.nn.rnn_cell.LSTMCell(config.num_hidden,state_is_tuple=True)
					self.lstm_outputs, self.states = rnn.rnn(self.lstm, self.splited_X_, dtype=tf.float32)
					self.lstm_out = self.lstm_outputs[-1]
				
				# *** DROPOUT ***
				with tf.name_scope('Dropout'):
					self.lstm_out_dropout = tf.nn.dropout(self.lstm_out, self.keep_prob)

				# *** OUTPUT LAYER ***
				with tf.name_scope('Output'):
					self.weights_out = tf.truncated_normal((config.num_hidden,config.vocabulary_size), stddev=0.1)
					self.biaises_out = tf.constant(0.1, shape=[config.vocabulary_size])

					self.logits_out = tf.matmul(self.lstm_out_dropout,self.weights_out) + self.biaises_out
					self.predicted_Y = tf.nn.softmax(self.logits_out)

			# *** LOSS ***
			with tf.name_scope('Loss'):
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits_out, self.Y_))

			# *** ACCURACY ***
			with tf.name_scope('Accuracy'):
				self.accuracy = tf.equal(tf.argmax(self.predicted_Y, 1), tf.argmax(self.Y_, 1))
				self.mean_accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
				
			# *** TRAIN STEP ***
			with tf.name_scope('Train_step'):
				self.train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

			# *** SUMMARIES ***
			tf.scalar_summary("loss", self.loss)
			tf.scalar_summary("mean_accuracy", self.mean_accuracy)
			self.merged_summary_op = tf.merge_all_summaries()

			# *** INITIALIZATION ***
			self.init = tf.initialize_all_variables()