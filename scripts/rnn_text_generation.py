"""
Text Generation (character level) using RNN

Inspired from http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

I reimplemented the model with tensorflow
"""

from __future__ import print_function
import time
import codecs
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

import model

# === CONSTANTS ===
# Paths
DATA_FOLDER = '../data/'
DATA_PATH = DATA_FOLDER + 'wonderland.txt'

# To clean the vocabulary
UNKNOWN_TOKEN = 'UKN'
# If empty, all characters are in the vocabulary and UNKNOWN_TOKEN is not used. Otherwise, replace those charcacters with UNKNOWN_TOKEN.
UNKNOWN_CHARS = ['\x80', '\x98', '\x99', '\x9c', '\x9d', '\xbb', '\xbf', '\xe2', '\xef']

# For training
LOGS_PATH = '../logs/'
NUM_EPOCHS = 2
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
training_config = model.TrainingConfig(vocabulary_size)
training_model = model.Model(training_config)

# === TRAINING ===
# Helper to display time
def seconds2minutes(time):
	minutes = int(time) / 60
	seconds = int(time) % 60
	return minutes, seconds

with tf.Session() as session:
	session.run(training_model.init)
	# op to write logs to Tensorboard
	summary_writer = tf.train.SummaryWriter(LOGS_PATH, graph=tf.get_default_graph())
	
	num_steps_per_epoch = num_sequences/BATCH_SIZE
	
	print("\nSTART TRAINING (",NUM_EPOCHS,"epochs,",num_steps_per_epoch,"steps per epoch )")
	begin_time = time_0 = time.time()
	
	for epoch in range(NUM_EPOCHS):
		print("*** EPOCH",epoch,"***")
		avg_loss = 0.0
		avg_accuracy = 0.0
		for step in range(num_steps_per_epoch):
			batch_X = X[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
			batch_Y = Y[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
			
			_, loss_value, accuracy_value, summary = session.run([training_model.train_step,training_model.loss,training_model.mean_accuracy,training_model.merged_summary_op], feed_dict={training_model.X_: batch_X, training_model.Y_: batch_Y, training_model.keep_prob: KEEP_PROB})
			
			avg_loss += loss_value
			avg_accuracy += accuracy_value
			
			# Write logs at every iteration
			absolute_step = epoch * num_steps_per_epoch + step
			summary_writer.add_summary(summary, absolute_step)
			
			if step % DISPLAY_STEP == 0:
				print("Batch Loss =",loss_value,"at step",absolute_step)
				print("Batch Accuracy =",accuracy_value,"at step",absolute_step)
				
				# Time spent is measured
				if absolute_step > 0:
					t = time.time()
					d = t - time_0
					time_0 = t
					
					print("Time:",d,"s to compute",DISPLAY_STEP,"steps")
			
		avg_loss = avg_loss/num_steps_per_epoch
		avg_accuracy = avg_accuracy/num_steps_per_epoch
		print("Average Batch Loss =",avg_loss,"at epoch",epoch)
		print("Average Batch Accuracy =",avg_accuracy,"at epoch",epoch)
	
	total_time = time.time() - begin_time
	total_time_minutes, total_time_seconds = seconds2minutes(total_time)
	print("*** Total time to compute",NUM_EPOCHS,"epochs:",total_time_minutes,"minutes and",total_time_seconds,"seconds (",total_time,"s)***")