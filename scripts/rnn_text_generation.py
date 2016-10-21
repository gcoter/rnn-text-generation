"""
Text Generation (character level) using RNN

Inspired from http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

I reimplemented the model with tensorflow
"""

from __future__ import print_function

import datamanager
import model
import constants
import train_utils
			
# === MAIN CODE ===
# *** CONSTRUCT DATASETS ***
X, Y, vocabulary_size = datamanager.create_datasets(constants.SEQ_LENGTH,constants.NUM_FEATURES)

# *** DEFINE MODEL ***
training_config = model.TrainingConfig(vocabulary_size)
training_model = model.Model(training_config)

# *** TRAINING ***
train_utils.train(training_model,X,Y,constants.NUM_EPOCHS,constants.BATCH_SIZE,constants.DISPLAY_STEP,constants.LOGS_PATH,constants.MODEL_PATH,constants.KEEP_PROB)