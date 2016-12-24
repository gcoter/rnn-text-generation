from __future__ import print_function
import time
import tensorflow as tf
import pickle

# Helper to display time
def seconds2minutes(time):
	minutes = int(time) // 60
	seconds = int(time) % 60
	return minutes, seconds
	
def test_on_valid_data(session,model,valid_X,valid_Y,batch_size):
	num_valid_sequences = len(valid_X)
	num_steps = num_valid_sequences//batch_size
	avg_loss = 0.0
	avg_accuracy = 0.0
	for step in range(num_steps):
		batch_X = valid_X[step * batch_size:(step + 1) * batch_size]
		batch_Y = valid_Y[step * batch_size:(step + 1) * batch_size]
		loss_value, accuracy_value = session.run([model.loss,model.mean_accuracy], feed_dict={model.X_: batch_X, model.Y_: batch_Y, model.keep_prob: 1.0})
		avg_loss += loss_value
		avg_accuracy += accuracy_value
	return avg_loss/num_steps, avg_accuracy/num_steps

def train(training_model,generation_model,train_X,train_Y,valid_X,valid_Y,num_epochs,batch_size,display_step,logs_path,model_path,keep_prob,char_to_int_dict=None,int_to_char_dict=None):
	if char_to_int_dict is None or int_to_char_dict is None:
		char_to_int_dict = pickle.load(open("../parameters/char_to_int_dict.pickle", "rb"))
		print("char_to_int_dict loaded")
		int_to_char_dict = pickle.load(open("../parameters/int_to_char_dict.pickle", "rb"))
		print("int_to_char_dict loaded")
	
	saver = tf.train.Saver()
	with tf.Session() as session:
		session.run(training_model.init)
		# op to write logs to Tensorboard
		# summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
		
		num_sequences = len(train_X)
		
		num_steps_per_epoch = num_sequences//batch_size
		
		print("\nSTART TRAINING (",num_epochs,"epochs,",num_steps_per_epoch,"steps per epoch )")
		begin_time = time_0 = time.time()
		
		for epoch in range(num_epochs):
			print("*** EPOCH",epoch,"***")
			avg_loss = 0.0
			avg_accuracy = 0.0
			for step in range(num_steps_per_epoch):
				batch_X = train_X[step * batch_size:(step + 1) * batch_size]
				batch_Y = train_Y[step * batch_size:(step + 1) * batch_size]
				
				_ = session.run(training_model.train_step, feed_dict={training_model.X_: batch_X, training_model.Y_: batch_Y, training_model.keep_prob: keep_prob})
				
				# Write logs at every iteration
				absolute_step = epoch * num_steps_per_epoch + step
				# summary_writer.add_summary(summary, absolute_step)
				
				if step % display_step == 0:
					loss_value, accuracy_value = session.run([training_model.loss,training_model.mean_accuracy], feed_dict={training_model.X_: batch_X, training_model.Y_: batch_Y, training_model.keep_prob: 1.0})
					print("Batch Loss =",loss_value,"at step",absolute_step)
					#print("Batch Accuracy =",accuracy_value,"at step",absolute_step)
					
					if step % (5*display_step) == 0:
						
						valid_loss, valid_accuracy = test_on_valid_data(session,training_model,valid_X,valid_Y,batch_size)
						print("Validation Loss =",valid_loss,"at step",absolute_step)
						#print("Validation Accuracy =",valid_accuracy,"at step",absolute_step)
						
						# Generate some text
						generated_text = generation_model.generate_text(session,char_to_int_dict,int_to_char_dict,text_size=100)
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
		
		total_time = time.time() - begin_time
		total_time_minutes, total_time_seconds = seconds2minutes(total_time)
		print("*** Total time to compute",num_epochs,"epochs:",total_time_minutes,"minutes and",total_time_seconds,"seconds (",total_time,"s)***")
		
		# Save parameters
		saver.save(session, model_path)
		print("Parameters saved to",model_path)