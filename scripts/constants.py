# === CONSTANTS ===
# Paths
DATA_FOLDER = '../data/'
DATA_PATH = DATA_FOLDER + 'wonderland.txt'
CHAR_TO_INT_PATH = DATA_FOLDER + "char_to_int_dict.pickle"
INT_TO_CHAR_PATH = DATA_FOLDER + "int_to_char_dict.pickle"

PARAMETERS_FOLDER = '../parameters/'
MODEL_PATH = PARAMETERS_FOLDER + 'model.ckpt'

# To clean the vocabulary
UNKNOWN_TOKEN = 'UKN'

# If empty, all characters are in the vocabulary and UNKNOWN_TOKEN is not used. Otherwise, replace those charcacters with UNKNOWN_TOKEN.
UNKNOWN_CHARS = [u'*',u'-',u'0',u'3',u'_',u'\u2018',u'\u2019',u'\u201c',u'\u201d',u'\ufeff'] #['*','-','0','3','_','‘','’','“', '”','\ufeff']

# Model parameters
NUM_HIDDEN = 512
NUM_RNN = 1
LEARNING_RATE = 1e-2
BATCH_SIZE = 64
SEQ_LENGTH = 100

# For training
NUM_EPOCHS = 10
KEEP_PROB = 0.5
DISPLAY_STEP = 100