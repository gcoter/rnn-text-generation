# === CONSTANTS ===
# Paths
DATA_FOLDER = '../data/'
DATA_PATH = DATA_FOLDER + 'wonderland.txt'
MODEL_PATH = '../parameters/model.ckpt'

# To clean the vocabulary
UNKNOWN_TOKEN = 'UKN'

# If empty, all characters are in the vocabulary and UNKNOWN_TOKEN is not used. Otherwise, replace those charcacters with UNKNOWN_TOKEN.
UNKNOWN_CHARS = ['\x80', '\x98', '\x99', '\x9c', '\x9d', '\xbb', '\xbf', '\xe2', '\xef']

# Model parameters
NUM_FEATURES = 1
SEQ_LENGTH = 100
BATCH_SIZE = 128
NUM_HIDDEN = 256
LEARNING_RATE = 1e-3

# For training
LOGS_PATH = '../logs/'
NUM_EPOCHS = 10
KEEP_PROB = 0.5
DISPLAY_STEP = 100