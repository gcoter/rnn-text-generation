from __future__ import print_function
import numpy as np
import pickle
import codecs
import os.path

import constants

class DataManager(object):
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
	
	def __init__(self,data_path,unknown_chars,unknown_token):
		self.data_path = data_path
		self.unknown_chars = unknown_chars
		self.unknown_token = unknown_token
		self.char_to_int_dict = None
		self.int_to_char_dict = None
		self.vocabulary_size = None
		self.train_X = None
		self.train_Y = None
		self.valid_X = None
		self.valid_Y = None
		
	def get_raw_text(self):
		with codecs.open(self.data_path, encoding='utf-8') as textfile:
			raw_text = textfile.read()
		raw_text = raw_text.lower() # Convert to lower case to reduce the vocabulary used (characters)
		return raw_text
		
	def save_dictionaries(self,char_to_int_path,int_to_char_path):
		DataManager.save_object(self.get_char_to_int_dict(),char_to_int_path,object_name="char_to_int_dict")
		DataManager.save_object(self.get_int_to_char_dict(),int_to_char_path,object_name="int_to_char_dict")
		
	def load_dictionaries(self,char_to_int_path,int_to_char_path):
		char_to_int_dict = DataManager.load_object(char_to_int_path,object_name="char_to_int_dict")
		int_to_char_dict = DataManager.load_object(int_to_char_path,object_name="int_to_char_dict")
		self.char_to_int_dict = char_to_int_dict
		self.int_to_char_dict = int_to_char_dict
		
	def get_dictionaries(self,char_to_int_path=None,int_to_char_path=None):
		return self.char_to_int_dict, self.int_to_char_dict
		
	def get_char_to_int_dict(self):
		return self.char_to_int_dict
		
	def get_int_to_char_dict(self):
		return self.int_to_char_dict

	def create_vocabulary(self,raw_text,char_to_int_path,int_to_char_path):
		chars = sorted(list(set(raw_text))) # This is the vocabulary

		# If unknown_chars is not empty...
		if self.unknown_chars:
			# ... clean vocabulary
			for unknown_char in self.unknown_chars:
				chars.remove(unknown_char)
			chars.append(self.unknown_token)

		self.char_to_int_dict = dict((char, index) for index, char in enumerate(chars)) # Mapping from char to int
		self.int_to_char_dict = dict((index, char) for index, char in enumerate(chars)) # Mapping from int to char
		
		# Save dictionaries
		self.save_dictionaries(char_to_int_path,int_to_char_path)
		
		return chars
	
	def char_to_int(self,char):
		char_to_int_dict = self.get_char_to_int_dict()
		if not char in char_to_int_dict.keys():
			return char_to_int_dict[self.unknown_token]
		else:
			return char_to_int_dict[char]
			
	def int_to_char(self,i):
		int_to_char_dict = self.get_int_to_char_dict()
		if not i in int_to_char_dict.keys():
			return self.unknown_token
		else:
			return int_to_char_dict[i]

	def X_to_categorical(self,dataX):
		vocabulary_size = self.get_vocabulary_size()
		data_np = np.array(dataX)
		return (np.arange(vocabulary_size) == data_np[:,:,None]).astype(int)
	
	def Y_to_categorical(self,dataY):
		vocabulary_size = self.get_vocabulary_size()
		data_np = np.array(dataY)
		res = np.zeros((len(data_np), vocabulary_size), dtype=np.int8)
		res[np.arange(len(data_np)),data_np] = 1
		return res
	
	def construct_datasets(self,char_to_int_path,int_to_char_path,seq_length,valid_proportion):
		raw_text = self.get_raw_text()
		chars = self.create_vocabulary(raw_text,char_to_int_path,int_to_char_path)
		
		raw_text_size = len(raw_text)
		self.vocabulary_size = len(chars)

		print("Vocabulary :",chars)
		print(self.vocabulary_size,"characters in vocabulary")

		dataX = [] # Sequences of characters (converted to int)
		dataY = [] # Character to predict from sequences (converted to int)
		for i in range(0, raw_text_size - seq_length):
			input_seq = raw_text[i:i + seq_length]
			char_out = raw_text[i + seq_length]
			dataX.append([self.char_to_int(char) for char in input_seq])
			dataY.append(self.char_to_int(char_out))

		num_sequences = len(dataX)
		print("Total number of sequences in dataset: ", num_sequences)

		# one hot encode the input variable
		X = self.X_to_categorical(dataX)
		# one hot encode the output variable
		Y = self.Y_to_categorical(dataY)
		
		print("X:",X.shape)
		print("Y:",Y.shape)
		
		print("\n*** Print some examples ***")
		i = 0
		print("=== EXAMPLE",i,"===")
		print("X[0]:")
		print(X[0])
		print("Y[0]:")
		print(Y[0])
		
		valid_index = int(len(X) * valid_proportion)
		self.valid_X = X[:valid_index]
		self.train_X = X[valid_index:]
		self.valid_Y = Y[:valid_index]
		self.train_Y = Y[valid_index:]
		
		return self.train_X,self.train_Y,self.valid_X,self.valid_Y
		
	def get_vocabulary_size(self):
		if not self.vocabulary_size is None:
			return self.vocabulary_size
		char_to_int_dict = self.get_char_to_int_dict()
		if not char_to_int_dict is None:
			return len(char_to_int_dict.keys())
		print("WARNING: vocabulary_size is None")
		return None
	
	def get_datasets(self,char_to_int_path=None,int_to_char_path=None,seq_length=None,valid_proportion=None):
		if self.train_X is None or self.train_Y is None or self.valid_X is None or self.valid_Y is None:
			if not char_to_int_path is None and not int_to_char_path is None and not seq_length is None and not valid_proportion is None:
				return self.construct_datasets(char_to_int_path,int_to_char_path,seq_length,valid_proportion)
			else:
				print("ERROR: please provide all parameters to construct the datasets.")
				print("get_datasets(self,char_to_int_path=None,int_to_char_path=None,seq_length=None,valid_proportion=None)")
				return self.train_X,self.train_Y,self.valid_X,self.valid_Y
		else:
			return self.train_X,self.train_Y,self.valid_X,self.valid_Y