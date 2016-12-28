from __future__ import print_function
import argparse

import constants
from datamanager import DataManager
from model import GenerationModel
from trainer import Trainer

def train(datamanager,args):
	print("============= CREATING DATASETS =============")
	train_X, train_Y, valid_X, valid_Y = datamanager.get_datasets(args.char_to_int_path,args.int_to_char_path,args.seq_length,args.valid_proportion)
	print()
	print("============= CREATING MODEL =============")
	generation_model = GenerationModel(datamanager.get_vocabulary_size(),args.num_hidden,args.num_rnn,args.learning_rate,args.batch_size,args.seq_length)
	print()
	print("============= TRAINING =============")
	trainer = Trainer(datamanager,generation_model)
	trainer.train(num_epochs=args.num_epochs,batch_size=args.batch_size,display_step=args.display_step,model_path=args.model_path,keep_prob=args.keep_prob)
	
def generate(datamanager,args):
	import tensorflow as tf
	print("============= LOADING DICTIONARIES =============")
	datamanager.load_dictionaries(args.char_to_int_path,args.int_to_char_path)
	print()
	print("============= CREATING MODEL =============")
	generation_model = GenerationModel(datamanager.get_vocabulary_size(),args.num_hidden,args.num_rnn,args.learning_rate,args.batch_size,args.seq_length)
	print()
	with tf.Session() as session:
		# Restore variables from disk
		print("============= RESTORING MODEL =============")
		saver = tf.train.Saver() 
		saver.restore(session, args.model_path)
		print("Model restored from file " + args.model_path)
		print()
		print("============= TEXT GENERATION =============")
		generated_text = generation_model.generate(datamanager,session,seed=args.seed,size=args.size,temperature=args.temperature)
		print("**************")
		print("GENERATED TEXT")
		print("**************")
		print(args.seed)
		print("**************")
		print(generated_text)
		print("**************")

# MAIN PARSER
parser = argparse.ArgumentParser(description="Text generation tool.")
parser.add_argument("--num_hidden",default=constants.NUM_HIDDEN,type=int,required=False,help="Number of hidden units in each RNN cell.")
parser.add_argument("--num_rnn",default=constants.NUM_RNN,type=int,required=False,help="Number of RNN cells.")
parser.add_argument("--learning_rate",default=constants.LEARNING_RATE,type=int,required=False,help="Learning rate used for training.")
parser.add_argument("--batch_size",default=constants.BATCH_SIZE,type=int,required=False,help="Batch size.")
parser.add_argument("--seq_length",default=constants.SEQ_LENGTH,type=int,required=False,help="Length of the sequences fed to the RNN.")
parser.add_argument("--model_path",default=constants.MODEL_PATH,type=str,required=False,help="Path to the model checkpoint file.")
# parser.add_argument("--unknown_chars",nargs='*',default=constants.UNKNOWN_CHARS,type=str,required=False,help="List of characters which should be ignored and replaced with UNKNOWN_TOKEN.")
parser.add_argument("--unknown_token",default=constants.UNKNOWN_TOKEN,type=str,required=False,help="Token used to replace unknown characters (defined in constants.py).")
parser.add_argument("--char_to_int_path",default=constants.CHAR_TO_INT_PATH,type=str,required=False,help="Path used to save and load the characters-to-integers dictionary.")
parser.add_argument("--int_to_char_path",default=constants.INT_TO_CHAR_PATH,type=str,required=False,help="Path used to save and load the integers-to-characters dictionary.")
parser.add_argument("--data_path",default=constants.DATA_PATH,type=str,required=False,help="Path to the data folder.")

# SUBPARSERS
subparsers = parser.add_subparsers(help="You can train the RNN or generate text with an already trained network.",dest="command")

# TRAINING SUBPARSER
train_parser = subparsers.add_parser("train", help="Train a new RNN.")
train_parser.add_argument("--valid_proportion",default=0.1,type=float,required=False,help="Proportion of examples used to construct the validation set.")
train_parser.add_argument("--num_epochs",default=constants.NUM_EPOCHS,type=int,required=False,help="Number of epochs to train the model.")
train_parser.add_argument("--display_step",default=constants.DISPLAY_STEP,type=int,required=False,help="The program displays validation loss and generated text every display_step steps.")
train_parser.add_argument("--keep_prob",default=constants.KEEP_PROB,type=int,required=False,help="Probability used for dropout.")

# GENERATION SUBPARSER
generate_parser = subparsers.add_parser("generate", help="Generate text with an already trained network.")
generate_parser.add_argument("--seed",default="Alice was beginning to get very tired of sitting by her sister on the\nbank, and of having nothing t",type=str,required=False,help="Seed used to generate text.")
generate_parser.add_argument("--size",default=constants.SEQ_LENGTH,type=int,required=False,help="Number of characters to be generated.")
generate_parser.add_argument("--temperature",default=1.0,type=float,required=False,help="Temperature used to generate text.")

# PARSE ARGUMENTS
args = parser.parse_args()
args_dict = vars(args)

print("============= ARGUMENTS =============")
for arg in args_dict.keys():
	print(arg,"->",args_dict[arg])
print()

datamanager = DataManager(args.data_path,constants.UNKNOWN_CHARS,args.unknown_token)

command = args.command
if command == "train":
	train(datamanager,args)
elif command == "generate":
	generate(datamanager,args)