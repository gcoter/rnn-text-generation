from __future__ import print_function
import time
import tensorflow as tf

import constants

class Trainer(object):
	def __init__(self,generation_model,train_X,train_Y,valid_X,valid_Y):
		self.generation_model = generation_model
		self.train_X = train_X
		self.train_Y = train_Y
		self.valid_X = valid_X
		self.valid_Y = valid_Y
		
	# Helper to display time
	@staticmethod
	def seconds2minutes(time):
		minutes = int(time) // 60
		seconds = int(time) % 60
		return minutes, seconds
		
	def test_on_valid_data(self,session,batch_size):
		num_valid_sequences = len(self.valid_X)
		num_steps = num_valid_sequences//batch_size
		avg_loss = 0.0
		for step in range(num_steps):
			batch_X = self.valid_X[step * batch_size:(step + 1) * batch_size]
			batch_Y = self.valid_Y[step * batch_size:(step + 1) * batch_size]
			loss_value = self.generation_model.get_loss(session,batch_X,batch_Y)
			avg_loss += loss_value
		return avg_loss/num_steps

	def train(self,num_epochs=constants.NUM_EPOCHS,batch_size=constants.BATCH_SIZE,display_step=constants.DISPLAY_STEP,model_path=constants.MODEL_PATH,keep_prob=constants.KEEP_PROB):
		saver = tf.train.Saver()
		with tf.Session() as session:
			self.generation_model.init(session,train=True)		
			num_sequences = len(self.train_X)
			num_steps_per_epoch = num_sequences//batch_size
			
			print("\nSTART TRAINING (",num_epochs,"epochs,",num_steps_per_epoch,"steps per epoch )")
			begin_time = time_0 = time.time()
			
			for epoch in range(num_epochs):
				print("*** EPOCH",epoch,"***")
				avg_loss = 0.0
				avg_accuracy = 0.0
				for step in range(num_steps_per_epoch):
					batch_X = self.train_X[step * batch_size:(step + 1) * batch_size]
					batch_Y = self.train_Y[step * batch_size:(step + 1) * batch_size]
					# Train step
					_ = self.generation_model.train_step(session,batch_X,batch_Y,keep_prob)
					absolute_step = epoch * num_steps_per_epoch + step
					# Display loss
					if step % display_step == 0:
						loss_value = self.generation_model.get_loss(session,batch_X,batch_Y)
						print("Batch Loss =",loss_value,"at step",absolute_step)
						# Test on validation data and generation
						if step % (5*display_step) == 0:
							valid_loss = self.test_on_valid_data(session,batch_size)
							print("Validation Loss =",valid_loss,"at step",absolute_step)
							# Generate some text
							seed = "Alice was beginning to get very tired of sitting by her sister on the\nbank, and of having nothing t"
							generated_text = seed + self.generation_model.generate(session,seed=seed,size=100,temperature=1.0)
							print("**************")
							print("GENERATED TEXT")
							print("**************")
							print(generated_text)
							print("**************\n")
						# Time spent is measured
						if absolute_step > 0:
							t = time.time()
							d = t - time_0
							time_0 = t
							print("Time:",d,"s to compute",display_step,"steps")
				
				# Save parameters
				saver.save(session, model_path)
				print("Parameters saved to",model_path)
				
			total_time = time.time() - begin_time
			total_time_minutes, total_time_seconds = Trainer.seconds2minutes(total_time)
			print("*** Total time to compute",num_epochs,"epochs:",total_time_minutes,"minutes and",total_time_seconds,"seconds (",total_time,"s)***")