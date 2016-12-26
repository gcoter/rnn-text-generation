from __future__ import print_function
import numpy as np
import pickle
import codecs
import os.path

import constants

class DataManager(object):
	char_to_int_dict = None
	int_to_char_dict = None

	@staticmethod
	def get_raw_text(data_path):
		with codecs.open(data_path, encoding='utf-8') as textfile:
			raw_text = textfile.read()
		raw_text = raw_text.lower() # Convert to lower case to reduce the vocabulary used (characters)
		return raw_text
		
	@staticmethod
	def save_object(object,save_path,object_name="Object"):
		pickle.dump(object, open(save_path, "wb"))
		print(object_name,'saved to',save_path)
			
	@staticmethod
	def load_object(load_path,object_name="Object"):
		if os.path.isfile(load_path):
			object = pickle.load(open(load_path, "rb"))
			print(object_name,'loaded from',load_path)
			return object
		else:
			print("ERROR: Unable to load",object_name,"from",load_path)
		
	@staticmethod
	def save_dictionaries(char_to_int_path=constants.CHAR_TO_INT_PATH,int_to_char_path=constants.INT_TO_CHAR_PATH):
		DataManager.save_object(DataManager.get_char_to_int_dict(),char_to_int_path,object_name="char_to_int_dict")
		DataManager.save_object(DataManager.get_int_to_char_dict(),int_to_char_path,object_name="int_to_char_dict")
		
	@staticmethod
	def load_dictionaries(char_to_int_path=constants.CHAR_TO_INT_PATH,int_to_char_path=constants.INT_TO_CHAR_PATH):
		char_to_int_dict = DataManager.load_object(char_to_int_path,object_name="char_to_int_dict")
		int_to_char_dict = DataManager.load_object(int_to_char_path,object_name="int_to_char_dict")
		DataManager.char_to_int_dict = char_to_int_dict
		DataManager.int_to_char_dict = int_to_char_dict
		
	@staticmethod
	def get_char_to_int_dict():
		if DataManager.char_to_int_dict is None:
			print("WARNING: char_to_int_dict is None")
			print("Try loading both dictionaries...")
			DataManager.load_dictionaries()
		return DataManager.char_to_int_dict
		
	@staticmethod
	def get_int_to_char_dict():
		if DataManager.int_to_char_dict is None:
			print("WARNING: int_to_char_dict is None")
			print("Try loading both dictionaries...")
			DataManager.load_dictionaries()
		return DataManager.int_to_char_dict

	@staticmethod
	def create_vocabulary(raw_text,unknown_chars=constants.UNKNOWN_CHARS,unknown_token=constants.UNKNOWN_TOKEN):
		chars = sorted(list(set(raw_text))) # This is the vocabulary

		# If unknown_chars is not empty...
		if unknown_chars:
			# ... clean vocabulary
			for unknown_char in unknown_chars:
				chars.remove(unknown_char)
			chars.append(unknown_token)

		DataManager.char_to_int_dict = dict((char, index) for index, char in enumerate(chars)) # Mapping from char to int
		DataManager.int_to_char_dict = dict((index, char) for index, char in enumerate(chars)) # Mapping from int to char
		
		# Save dictionaries
		DataManager.save_dictionaries()
		
		return chars
	
	@staticmethod
	def char_to_int(char,unknown_token=constants.UNKNOWN_TOKEN):
		char_to_int_dict = DataManager.get_char_to_int_dict()
		if not char in char_to_int_dict.keys():
			return char_to_int_dict[unknown_token]
		else:
			return char_to_int_dict[char]
			
	@staticmethod
	def int_to_char(i,unknown_token=constants.UNKNOWN_TOKEN):
		int_to_char_dict = DataManager.get_int_to_char_dict()
		if not i in int_to_char_dict.keys():
			return unknown_token
		else:
			return int_to_char_dict[i]
	
	@staticmethod
	def to_categorical(data,vocabulary_size):
		data_np = np.array(data)
		res = np.zeros((len(data_np), vocabulary_size), dtype=np.int8)
		res[np.arange(len(data_np)),data_np] = 1
		return res
	
	@staticmethod
	def create_datasets(valid_proportion=0.1,data_path=constants.DATA_PATH,seq_length=constants.SEQ_LENGTH,num_features=constants.NUM_FEATURES):
		raw_text = DataManager.get_raw_text(data_path)
		chars = DataManager.create_vocabulary(raw_text)
		
		raw_text_size = len(raw_text)
		vocabulary_size = len(chars)

		print("Vocabulary :",chars)
		print(vocabulary_size,"characters in vocabulary")

		dataX = [] # Sequences of characters (converted to int)
		dataY = [] # Character to predict from sequences (converted to int)
		for i in range(0, raw_text_size - seq_length):
			input_seq = raw_text[i:i + seq_length]
			char_out = raw_text[i + seq_length]
			dataX.append([DataManager.char_to_int(char) for char in input_seq])
			dataY.append(DataManager.char_to_int(char_out))

		num_sequences = len(dataX)
		print("Total number of sequences in dataset: ", num_sequences)

		# reshape X to be [samples, time steps, number of features]
		X = np.reshape(dataX, (num_sequences, seq_length, num_features))
		# one hot encode the output variable
		Y = DataManager.to_categorical(dataY,vocabulary_size)
		
		print("\nPrint some examples")
		
		for i in [0,100,200,400,500]:
			print("\n=== EXAMPLE",i,"===")
			x = ''.join([DataManager.int_to_char(c[0]) for c in X[i]])
			y = DataManager.int_to_char(np.argmax(Y[i]))
			print(x,"-->",y)
		print()
		
		valid_index = int(len(X) * valid_proportion)
		valid_X = X[:valid_index]
		train_X = X[valid_index:]
		valid_Y = Y[:valid_index]
		train_Y = Y[valid_index:]
		
		return train_X,train_Y,valid_X,valid_Y
		
	@staticmethod
	def get_vocabulary_size():
		char_to_int_dict = DataManager.get_char_to_int_dict()
		if not char_to_int_dict is None:
			return len(char_to_int_dict.keys())
		return None
	
	@staticmethod
	def get_datasets():
		return DataManager.create_datasets()