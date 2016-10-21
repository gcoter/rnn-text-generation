from __future__ import print_function
import numpy as np
# import codecs

import constants

def get_raw_text(data_path):
	#with codecs.open(data_path, encoding='utf-8') as textfile:
	with open(data_path, 'r') as textfile:
		raw_text = textfile.read()

	raw_text = raw_text.lower() # Convert to lower case to reduce the vocabulary used (characters)
	return raw_text

def create_vocabulary(raw_text):
	chars = sorted(list(set(raw_text))) # This is the vocabulary

	# If UNKNOWN_CHARS is not empty...
	if constants.UNKNOWN_CHARS:
		# ... clean vocabulary
		for unknown_char in constants.UNKNOWN_CHARS:
			chars.remove(unknown_char)
		chars.append(constants.UNKNOWN_TOKEN)

	char_to_int_dict = dict((char, index) for index, char in enumerate(chars)) # Mapping from char to int
	int_to_char_dict = dict((index, char) for index, char in enumerate(chars)) # Mapping from int to char
	
	return chars, char_to_int_dict, int_to_char_dict

def char_to_int(char_to_int_dict,char):
	if constants.UNKNOWN_CHARS and char in constants.UNKNOWN_CHARS:
		return char_to_int_dict[constants.UNKNOWN_TOKEN]
	else:
		return char_to_int_dict[char]
		
def int_to_char(int_to_char_dict,char):
	if constants.UNKNOWN_CHARS and char in constants.UNKNOWN_CHARS:
		return int_to_char_dict[constants.UNKNOWN_TOKEN]
	else:
		return int_to_char_dict[char]

def to_categorical(data,vocabulary_size):
	data_np = np.array(data)
	res = np.zeros((len(data_np), vocabulary_size), dtype=np.bool)
	res[np.arange(len(data_np)),data_np] = 1
	return res
		
def create_datasets(seq_length,num_features):
	raw_text = get_raw_text(constants.DATA_PATH)
	chars, char_to_int_dict, int_to_char_dict = create_vocabulary(raw_text)
	
	raw_text_size = len(raw_text)
	vocabulary_size = len(chars)

	print("Vocabulary :",chars)
	print(vocabulary_size,"characters in vocabulary")

	dataX = [] # Sequences of characters (converted to int)
	dataY = [] # Character to predict from sequences (converted to int)
	for i in range(0, raw_text_size - seq_length):
		input_seq = raw_text[i:i + seq_length]
		char_out = raw_text[i + seq_length]
		dataX.append([char_to_int(char_to_int_dict,char) for char in input_seq])
		dataY.append(char_to_int(char_to_int_dict,char_out))

	num_sequences = len(dataX)
	print("Total number of sequences in dataset: ", num_sequences)

	# reshape X to be [samples, time steps, number of features]
	X = np.reshape(dataX, (num_sequences, seq_length, num_features))
	# normalize
	X = X / float(vocabulary_size)
	# one hot encode the output variable
	Y = to_categorical(dataY,vocabulary_size)

	print("X shape:",X.shape)
	print("Y shape:",Y.shape)
	
	return X,Y,vocabulary_size