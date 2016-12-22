"""
Text Generation (character level) using RNN

Inspired from http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

I reimplemented the model with tensorflow
"""

from __future__ import print_function
import os.path
import tensorflow as tf
import numpy as np
import random
import pickle

import datamanager
import model
import constants
import train_utils

def sample(probabilities, temperature=1.0):
	probabilities = np.log(probabilities) / temperature
	probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
	r = random.random() # range: [0,1)
	total = 0.0
	for i in range(len(probabilities)):
		total += probabilities[i]
		if total>r:
			return i
	return len(probabilities)-1 

# === MAIN CODE ===
if not os.path.isfile(constants.MODEL_PATH):
	print(constants.MODEL_PATH,"not found : start training procedure...")

	# *** CONSTRUCT DATASETS ***
	X, Y, vocabulary_size = datamanager.create_datasets(constants.SEQ_LENGTH,constants.NUM_FEATURES)
	valid_index = int(len(X) * 0.1)

	valid_X = X[:valid_index]
	train_X = X[valid_index:]
	valid_Y = Y[:valid_index]
	train_Y = Y[valid_index:]
	
	print("train_X:",train_X.shape)
	print("train_Y:",train_Y.shape)
	print("valid_X:",valid_X.shape)
	print("valid_Y:",valid_Y.shape)

	# *** DEFINE MODEL ***
	training_config = model.TrainingConfig(vocabulary_size)
	training_model = model.Model(training_config)

	# *** TRAINING ***
	train_utils.train(training_model,train_X,train_Y,valid_X,valid_Y,constants.NUM_EPOCHS,constants.BATCH_SIZE,constants.DISPLAY_STEP,constants.LOGS_PATH,constants.MODEL_PATH,constants.KEEP_PROB)
else:
	print(constants.MODEL_PATH,"found : start text generation...")
	
	# *** LOAD VOCABULARY ***
	char_to_int_dict = pickle.load(open("../parameters/char_to_int_dict.pickle", "rb"))
	print("char_to_int_dict loaded")
	int_to_char_dict = pickle.load(open("../parameters/int_to_char_dict.pickle", "rb"))
	print("int_to_char_dict loaded")
	
	vocabulary_size = len(char_to_int_dict)
	
	# *** DEFINE MODEL ***
	generation_config = model.GenerationConfig(vocabulary_size)
	generation_model = model.Model(generation_config)
	
	with tf.Session() as session:
		# Restore variables from disk
		saver = tf.train.Saver() 
		saver.restore(session, constants.MODEL_PATH)
		print("Model restored from file " + constants.MODEL_PATH)
		
		# *** Generate Text ***
		text_size = 100
		generated_text = ""
		states = generation_model.initial_state.eval()
		
		input = np.array([[[datamanager.char_to_int(char_to_int_dict,".")]]])
		
		for i in range(text_size):
			probabilities, states = session.run([generation_model.predicted_Y,generation_model.states], feed_dict={generation_model.X_: input, generation_model.initial_state: states, generation_model.keep_prob: 1.0})
			generated_text_index = sample(probabilities[0], temperature=1.0)
			generated_character = datamanager.int_to_char(int_to_char_dict,generated_text_index)
			generated_text += generated_character
			input = np.array([[[generated_text_index]]])
			
		print("GENERATED TEXT")
		print()
		print(generated_text)