from __future__ import print_function

import constants
from datamanager import DataManager
from model import GenerationModel
from trainer import Trainer

train = True

if train:
	print("============= CREATING DATASETS =============")
	train_X, train_Y, valid_X, valid_Y = DataManager.create_datasets()
	print("============= CREATING MODEL =============")
	generation_model = GenerationModel(DataManager.get_vocabulary_size(),seq_length=constants.SEQ_LENGTH,batch_size=constants.BATCH_SIZE)
	print("============= TRAINING =============")
	trainer = Trainer(generation_model,train_X,train_Y,valid_X,valid_Y)
	trainer.train(num_epochs=10,batch_size=constants.BATCH_SIZE,display_step=constants.DISPLAY_STEP,model_path=constants.MODEL_PATH,keep_prob=constants.KEEP_PROB)
else:
	import tensorflow as tf
	generation_model = GenerationModel(DataManager.get_vocabulary_size(),seq_length=constants.SEQ_LENGTH,batch_size=constants.BATCH_SIZE)
	with tf.Session() as session:
		# Restore variables from disk
		saver = tf.train.Saver() 
		saver.restore(session, constants.MODEL_PATH)
		print("Model restored from file " + constants.MODEL_PATH)
		seed = "Alice was beginning to get very tired of sitting by her sister on the\nbank, and of having nothing t"
		generated_text = generation_model.generate(session,seed=seed,size=100,temperature=1.0)
		print("**************")
		print("GENERATED TEXT")
		print("**************")
		print(seed + generated_text)
		print("**************")

"""
print("X:",X.shape)
print("Y:",X.shape)
print("\nTen first examples\n")
print("X[0:10]:\n",X[0:10])
print("Y[0:10]:\n",Y[0:10])
"""