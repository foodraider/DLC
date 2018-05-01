'''
 THis is the script to train our model
'''

import tensorflow as tf
from batch_generation import *
from chatbot import ChatbotModel
import math
import os

##################### Control Panel ###########################

hidden_size = 1024
num_hidden_layers = 2
word_embedding_size = 1024
encoder_dropout_rate_placeholder = 0.5
Train = True
Decode = False
beam_size = 1
learning_rate = 0.0001
numEpochs = 100
batch_size = 128
steps_per_checkpoint = 50
load_from_prev = True
use_beam_search = False

##################### Control Panel ###########################

print ("I am using tensorflow version of: ")
print (str(tf.__version__))

log_dir = '' # directory to save the log information (for tensorboard)
data_store = '' # directory for the dataset
file_name = '' # name of the data file
dataset_path = os.path.join(data_store, file_name)
model_dir_path = ""  # directory where the model is saved
model_name = "" # name of the model to be saved

word2id, id2word, training_data = loadData(dataset_path)
total_size = len(training_data)
print('The training size is of %d ' % total_size)
vocab_size = len(word2id)
print('The vocab size is of %d ' % vocab_size)

with tf.Session() as sess:
    model = ChatbotModel(hidden_size, num_hidden_layers, vocab_size, word_embedding_size, encoder_dropout_rate_placeholder,
                 Train, Decode, use_beam_search, beam_size, word2id, learning_rate)

    ckpt = tf.train.get_checkpoint_state(model_dir_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and load_from_prev:
        print('Loading parameters from previously trained model')
        print('==============================')
        print(ckpt.model_checkpoint_path)
        print('==============================')

        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Initialize a new model')
        sess.run(tf.global_variables_initializer())
    current_step = 0
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    for e in range(numEpochs):
        print("============= Training Epoch:{} =========== ".format(e + 1))
        batches = generateBatches(training_data, batch_size)
        for c, nextBatch in enumerate(batches):
            loss = model.train(sess, nextBatch)
            current_step += 1
            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print ("Iteration: %d. Loss %.2f. Perplexity %.2f" % (current_step, loss, perplexity))
                checkpoint_path = os.path.join(model_dir_path, model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)