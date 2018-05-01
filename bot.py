"""

    This script uses the Telegram API platform to create a Chatbot

    ####################################################################################################################
    Bot name: Trinity Bot
    username: trinity_mecc_bot
    address: http://t.me/trinity_mecc_bot
    token:
    ####################################################################################################################

"""


import json
import requests
import time




import tensorflow as tf
from batch_generation import *
from chatbot import ChatbotModel
import math
import os
import numpy as np
import sys

##################### Control Panel ###########################

hidden_size = 1024
num_hidden_layers = 2
word_embedding_size = 1024
encoder_dropout_rate_placeholder = 0.0
Train = False
Decode = True
beam_size = 1
learning_rate = 0.0001
numEpochs = 30
batch_size = 128
steps_per_checkpoint = 2
load_from_prev = True
use_beam_search = True
model_dir_path = ""

##################### Control Panel ###########################


deleteConvAfterEnd = True # We do not keep a history of all conversations
TOKEN = ""
URL = "https://api.telegram.org/bot{}/".format(TOKEN)


def get_url(url):
    """
    This function takes in an url and return the content object, decoded
    :param url: the url of the http command
    :return: content returned by the http request
    """
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    """
    This function takes in an url and return the json-formatted content
    :param url: the url of the http command
    :return: the json format of content in that URL
    """
    content = get_url(url)
    js = json.loads(content)
    return js

def get_updates(offset=None):
    """
    This is our function to use the API get update
    :return: the json format of content in that URL page (which is the update)
    """
    url = URL + "getUpdates"
    if offset:
        url += "?offset={}".format(offset)
    js = get_json_from_url(url)
    return js

def get_last_update_id(updates):
    """
    This is a function that keeps all the update IDs and return the highest (most recent one)
    :param updates: events returned from http request
    :return: the most recent event that we would like to respond to
    """
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return update_ids[-1]

def get_last_chat_id_and_text(updates):
    """
    This is our function to obtain the most recent update
    :param updates: the content from the getUpdate API, which consists of the recently received messages
    :return: the most recent text, and its associated chat ID
    """
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def create_response_string(msg, chat_ID, database, model, sess, word2id, id2word):
    """
    This function process a message from the user and gives a response in the format convAI required
    :param msg: user's msg
    :param score: our evaluation on the user's message
    :return: response, in the form of a string
    """
    response = {}
    if msg != "/end" and msg != "/begin":
        ##############################################
        batch = sentence2batch(msg, word2id)
        predicted_ids = model.inference(sess, batch)
        response = convert_ids_to_seq(predicted_ids, id2word, beam_size)
        ##############################################
    elif msg == "/end":
        response = "Nice talking to you"
    else:
        response = "Hey there!"
    return response

def send_message(msg, chat_id, database, model, sess, word2id, id2word):
    """
    This is our function to send message to others
    :param text: the message we want to respond to and evaluate
    :param chat_id: the user ID that we need to respond to
    :return: na
    """
    response = create_response_string(msg, chat_id, database, model, sess, word2id, id2word)
    url = URL + "sendMessage?text={}&chat_id={}".format(response, chat_id)
    get_url(url)
    database[chat_id]["DTH_hist"].append(response) #record our response to the user in the database

def begin_conv(chat_id, database):
    """
    :param chat_id: the user ID that we need to respond to
    :param database: the database storing the conversation
    :return:
    """
    msg = "/begin"
    response = create_response_string(msg, chat_id, database)
    url = URL + "sendMessage?text={}&chat_id={}".format(response, chat_id)
    get_url(url)
    database[chat_id]["DTH_hist"].append(msg) # record in DTH_history

def end_conv(chat_id, database):
    """
    :param chat_id: the user ID that we need to respond to
    :param database: the database storing the conversation
    :return:
    """
    msg = "/end"
    response = create_response_string(msg, chat_id, database)
    url = URL + "sendMessage?text={}&chat_id={}".format(response, chat_id)
    get_url(url)
    print ()
    print ("Checking my database, this conversation has ended: ")
    print (database[chat_id])
    print ("In the past conversation, user said the following sentences: ")
    print (database[chat_id]['user_hist'])
    print ("In the past conversation, out DeepTalkHawk Bot said the following sentences: ")
    print (database[chat_id]['DTH_hist'])
    print ()
    if deleteConvAfterEnd == True:
        del database[chat_id]


def convert_ids_to_seq(predict_ids, id2word, 1):
    """
    :param predict_ids: The response of our chatbot in integer word IDs
    :param id2word: the dictionary through which we convert id to word
    :param
    :return:
    """
    result = ""
    for single_predict in predict_ids:
        predict_list = np.ndarray.tolist(single_predict[:, :, 1])
        predict_seq = [id2word[idx] for idx in predict_list[0]]
        result = result + " ".join(predict_seq)
    return result


def process_update(text, chat_ID, database, model, sess, word2id, id2word):
    """
    This function makes appropriate responses to each particular update (text, chat_D) pair.
    :param text: the text from the user
    :param chat_ID: user ID
    :return: nothing
    """
    if text == "/begin":
        begin_conv(chat_ID, database)
    elif text == "/end" or text[:4] == "/end":
        end_conv(chat_ID, database)
    elif text[:6] == "/start":
        pass
    else:
        send_message(text, chat_ID, database, model, sess, word2id, id2word)


def process_database(text, chat_ID, database):
    """
    This function takes in the user's text and user's ID and make appropriate adjustment on the database for that user
    :param text: user's input message text
    :param chat_ID: user's ID
    :param database: the database we use to store history of conversations
    :return:
    """
    if not (chat_ID in database): # if this is a new user, initialize a field in the dictionary
        database[chat_ID] = {}
        database[chat_ID]["context"] = ""
        database[chat_ID]["user_hist"] = []
        database[chat_ID]["DTH_hist"] = []
        database[chat_ID]["whostart"] = "user"
    if text == "/begin":            # if our bot is asked to initialize a conversation
        database[chat_ID]["whostart"] = "DTH"
    elif text[:6] == "/start":      # when a context is given
        database[chat_ID]["context"] = text[7:]
    else:                           # in other cases, simply put the user's message text in the user_hist
        database[chat_ID]["user_hist"].append(text)

def main():
    print("Loading the model ... ")
    print ("Please go to http://t.me/trinity_mecc_bot to talk to me.")
    word2id, id2word, training_data = loadData(dataset_path)
    total_size = len(training_data)
    print('The training size is of %d ' % total_size)
    vocab_size = len(word2id)
    print('The vocab size is of %d ' % vocab_size)
    print ("Finished loading the model ... ")
    print ("")


    """ Initialize a tensorflow session for the conversation"""
    with tf.Session() as sess:
        model = ChatbotModel(hidden_size, num_hidden_layers, vocab_size, word_embedding_size,
                             encoder_dropout_rate_placeholder,
                             Train, Decode, use_beam_search, beam_size, word2id, learning_rate)
        ckpt = tf.train.get_checkpoint_state(model_dir_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and load_from_prev:
            print('Reloading model parameters..')
            print('==============================')
            print(ckpt.model_checkpoint_path)
            print('==============================')

            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Error: No Saved Model to Retrieve!')

        last_update_id = None
        database = {}
        while True:
            updates = get_updates(last_update_id)
            if len(updates["result"]) > 0:
                last_update_id = get_last_update_id(updates) + 1
                for update in updates["result"]:
                    try:
                        text = update["message"]["text"]

                        chat_ID = update["message"]["chat"]["id"]
                        process_database(text, chat_ID, database)
                        process_update(text, chat_ID, database, model, sess, word2id, id2word)
                    except Exception as e:
                        print ("Found error: ")
                        print (e)
            time.sleep(0.1)


if __name__ == '__main__':
    main()