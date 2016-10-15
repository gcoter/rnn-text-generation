"""
Text Generation (character level) using RNN

Inspired from http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

I reimplemented the model with tensorflow
"""

from __future__ import print_function
import codecs
import numpy as np
import tensorflow as tf

DATA_FOLDER = 'data/'
DATA_PATH = DATA_FOLDER + 'wonderland.txt'

# === GET THE DATA ===
#with codecs.open(DATA_PATH, encoding='utf-8') as textfile:
with open(DATA_PATH, 'r') as textfile:
	raw_text = textfile.read()

raw_text = raw_text.lower() # Convert to lower case to reduce the vocabulary used (characters)

# === CREATE VOCABULARY ===
chars = sorted(list(set(raw_text))) # This is the vocabulary
char_to_int = dict((char, index) for index, char in enumerate(chars)) # Mapping from char to int
int_to_char = dict((index, char) for index, char in enumerate(chars)) # Mapping from int to char

raw_text_size = len(raw_text)
vocabulary_size = len(chars)

print(vocabulary_size,"characters in vocabulary")

# === CREATE DATASETS ===
SEQ_LENGTH = 100

dataX = [] # Sequences of characters (converted to int)
dataY = [] # Character to predict from sequences (converted to int)
for i in range(0, raw_text_size - SEQ_LENGTH):
	input_seq = raw_text[i:i + SEQ_LENGTH]
	char_out = raw_text[i + SEQ_LENGTH]
	dataX.append([char_to_int[char] for char in input_seq])
	dataY.append(char_to_int[char_out])

num_sequences = len(dataX)
print("Total number of sequences in dataset: ", num_sequences)

# === PREPARE DATASETS ===
def to_categorical(data,vocabulary_size):
	data_np = np.array(data)
	res = np.zeros((len(data_np), vocabulary_size), dtype=np.bool)
	res[np.arange(len(data_np)),data_np] = 1
	return res

# reshape X to be [samples, time steps]
X = np.reshape(dataX, (num_sequences, SEQ_LENGTH))
# normalize
X = X / float(vocabulary_size)
# one hot encode the output variable
Y = to_categorical(dataY,vocabulary_size)

print("X shape:",X.shape)
print("Y shape:",Y.shape)

# === DEFINE MODEL ===
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_HIDDEN = 256

X_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE,SEQ_LENGTH))
Y_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE,vocabulary_size))

keep_prob = tf.placeholder(tf.float32)

# *** LSTM ***
with tf.name_scope('Model'):
	with tf.name_scope('LSTM'):
		lstm = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN,state_is_tuple=True)
		# Initial state of the LSTM memory
		state = lstm.zero_state(BATCH_SIZE,tf.float32)

		# The value of state is updated after processing each batch of characters
		lstm_out, state = lstm(X_, state)
	
	with tf.name_scope('Dropout'):
		# *** DROPOUT ***
		lstm_out_dropout = tf.nn.dropout(lstm_out, keep_prob)

	with tf.name_scope('Output'):
		# *** OUTPUT LAYER ***
		weights_out = tf.truncated_normal((NUM_HIDDEN,vocabulary_size), stddev=0.1)
		biaises_out = tf.constant(0.1, shape=[vocabulary_size])

		logits_out = tf.matmul(lstm_out_dropout,weights_out) + biaises_out
		predicted_Y = tf.nn.softmax(logits_out)

with tf.name_scope('Loss'):
	# *** LOSS ***
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_out, Y_))

with tf.name_scope('Train_step'):
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# === TRAINING ===
