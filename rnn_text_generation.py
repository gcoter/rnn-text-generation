"""
Text Generation (character level) using RNN

Inspired from http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

I reimplemented the model with tensorflow
"""

from __future__ import print_function
import codecs
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

# === CONSTANTS ===
# Paths
DATA_FOLDER = 'data/'
DATA_PATH = DATA_FOLDER + 'wonderland.txt'

# To clean the vocabulary
UNKNOWN_TOKEN = 'UKN'
# If empty, all characters are in the vocabulary and UNKNOWN_TOKEN is not used. Otherwise, replace those charcacters with UNKNOWN_TOKEN.
UNKNOWN_CHARS = ['\x80', '\x98', '\x99', '\x9c', '\x9d', '\xbb', '\xbf', '\xe2', '\xef']

# Model parameters
SEQ_LENGTH = 100
NUM_FEATURES = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_HIDDEN = 256

# For training
LOGS_PATH = 'logs/'
NUM_EPOCHS = 5
KEEP_PROB = 0.5
DISPLAY_STEP = 100

# === GET THE DATA ===
#with codecs.open(DATA_PATH, encoding='utf-8') as textfile:
with open(DATA_PATH, 'r') as textfile:
	raw_text = textfile.read()

raw_text = raw_text.lower() # Convert to lower case to reduce the vocabulary used (characters)

# === CREATE VOCABULARY ===
chars = sorted(list(set(raw_text))) # This is the vocabulary

# If UNKNOWN_CHARS is not empty...
if UNKNOWN_CHARS:
	# ... clean vocabulary
	for unknown_char in UNKNOWN_CHARS:
		chars.remove(unknown_char)
	chars.append(UNKNOWN_TOKEN)

char_to_int_dict = dict((char, index) for index, char in enumerate(chars)) # Mapping from char to int
int_to_char_dict = dict((index, char) for index, char in enumerate(chars)) # Mapping from int to char

def char_to_int(char):
	if UNKNOWN_CHARS and char in UNKNOWN_CHARS:
		return char_to_int_dict[UNKNOWN_TOKEN]
	else:
		return char_to_int_dict[char]
		
def int_to_char(char):
	if UNKNOWN_CHARS and char in UNKNOWN_CHARS:
		return char_to_int_dict[UNKNOWN_TOKEN]
	else:
		return int_to_char_dict[char]
		
raw_text_size = len(raw_text)
vocabulary_size = len(chars)

print("Vocabulary :",chars)
print(vocabulary_size,"characters in vocabulary")

# === CREATE DATASETS ===
dataX = [] # Sequences of characters (converted to int)
dataY = [] # Character to predict from sequences (converted to int)
for i in range(0, raw_text_size - SEQ_LENGTH):
	input_seq = raw_text[i:i + SEQ_LENGTH]
	char_out = raw_text[i + SEQ_LENGTH]
	dataX.append([char_to_int(char) for char in input_seq])
	dataY.append(char_to_int(char_out))

num_sequences = len(dataX)
print("Total number of sequences in dataset: ", num_sequences)

# === PREPARE DATASETS ===
def to_categorical(data,vocabulary_size):
	data_np = np.array(data)
	res = np.zeros((len(data_np), vocabulary_size), dtype=np.bool)
	res[np.arange(len(data_np)),data_np] = 1
	return res

# reshape X to be [samples, time steps, number of features]
X = np.reshape(dataX, (num_sequences, SEQ_LENGTH, NUM_FEATURES))
# normalize
X = X / float(vocabulary_size)
# one hot encode the output variable
Y = to_categorical(dataY,vocabulary_size)

print("X shape:",X.shape)
print("Y shape:",Y.shape)

# === DEFINE MODEL ===
with tf.name_scope('X_'):
	X_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE,SEQ_LENGTH,NUM_FEATURES))
	
	""" From https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py """
	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
	
	# Permuting batch_size and n_steps
	transposed_X_ = tf.transpose(X_, [1, 0, 2])
	# Reshaping to (n_steps*batch_size, n_input)
	reshaped_X_ = tf.reshape(transposed_X_, [-1, NUM_FEATURES])
	# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	splited_X_ = tf.split(0, SEQ_LENGTH, reshaped_X_)

with tf.name_scope('Y_'):
	Y_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE,vocabulary_size))

with tf.name_scope('keep_prob'):
	keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('Model'):
	# *** LSTM ***
	with tf.name_scope('LSTM'):
		lstm = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN,state_is_tuple=True)
		lstm_outputs, states = rnn.rnn(lstm, splited_X_, dtype=tf.float32)
		lstm_out = lstm_outputs[-1]
	
	# *** DROPOUT ***
	with tf.name_scope('Dropout'):
		lstm_out_dropout = tf.nn.dropout(lstm_out, keep_prob)

	# *** OUTPUT LAYER ***
	with tf.name_scope('Output'):
		weights_out = tf.truncated_normal((NUM_HIDDEN,vocabulary_size), stddev=0.1)
		biaises_out = tf.constant(0.1, shape=[vocabulary_size])

		logits_out = tf.matmul(lstm_out_dropout,weights_out) + biaises_out
		predicted_Y = tf.nn.softmax(logits_out)

# *** LOSS ***
with tf.name_scope('Loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_out, Y_))

# *** ACCURACY ***
with tf.name_scope('Accuracy'):
	accuracy = tf.equal(tf.argmax(predicted_Y, 1), tf.argmax(Y_, 1))
	mean_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
	
# *** TRAIN STEP ***
with tf.name_scope('Train_step'):
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# *** SUMMARIES ***
tf.scalar_summary("loss", loss)
tf.scalar_summary("mean_accuracy", mean_accuracy)
merged_summary_op = tf.merge_all_summaries()

# *** INITIALIZATION ***
init = tf.initialize_all_variables()

# === TRAINING ===
with tf.Session() as session:
	session.run(init)
	# op to write logs to Tensorboard
	summary_writer = tf.train.SummaryWriter(LOGS_PATH, graph=tf.get_default_graph())
	
	num_steps_per_epoch = num_sequences/BATCH_SIZE
	
	print("\nSTART TRAINING (",NUM_EPOCHS,"epochs,",num_steps_per_epoch,"steps per epoch )")
	for epoch in range(NUM_EPOCHS):
		print("*** EPOCH",epoch,"***")
		avg_loss = 0.0
		avg_accuracy = 0.0
		for step in range(num_steps_per_epoch):
			batch_X = X[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
			batch_Y = Y[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
			
			_, loss_value, accuracy_value, summary = session.run([train_step,loss,mean_accuracy,merged_summary_op], feed_dict={X_: batch_X, Y_: batch_Y, keep_prob: KEEP_PROB})
			
			avg_loss += loss_value
			avg_accuracy += accuracy_value
			
			# Write logs at every iteration
			summary_writer.add_summary(summary, epoch * num_steps_per_epoch + step)
			
			if step % DISPLAY_STEP == 0:
				print("Batch Loss =",loss_value,"at step",epoch * num_steps_per_epoch + step)
				print("Batch Accuracy =",accuracy_value,"at step",epoch * num_steps_per_epoch + step)
			
		avg_loss = avg_loss/num_steps_per_epoch
		avg_accuracy = avg_accuracy/num_steps_per_epoch
		print("Average Batch Loss =",avg_loss,"at epoch",epoch)
		print("Average Batch Accuracy =",avg_accuracy,"at epoch",epoch)