'''

    This script handles generating batches of messages - response pairs.

'''

import os
import pickle
import random

from batchStructure import Batch
import nltk
nltk.download('punkt')


# some macros
data_store = ''
file_name = ''
dataset_path = os.path.join(data_store, file_name)
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


def loadData(dataset_path):
    """
    This method loads the dataset
    :param dataset_path: the directory where our data is stored
    :return: the dictionaries (word2id, id2word) and conversations training_data
    """
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        word2id = data['word2id']
        id2word = data['id2word']
        training_data = data['trainingSamples']
    return word2id, id2word, training_data




def processBatch(unprocessed_batch):
    """

    :param unprocessed_batch: a batch of conversations, unpadded
    :return: a batch of conversations, all padded to become the same length
    """
    processed_batch = Batch()
    processed_batch.inputMsgLength = [len(pair[0]) for pair in unprocessed_batch]
    processed_batch.outputResponseLength = [len(pair[1]) for pair in unprocessed_batch]

    max_input_length = max(processed_batch.inputMsgLength)
    max_output_length = max(processed_batch.outputResponseLength)

    for pair in unprocessed_batch:
        msg = list(reversed(pair[0]))
        # we pad all input into equal length
        processed_batch.inputMsgIDs.append([PAD_TOKEN] * (max_input_length - len(msg)) + msg)
        resp = list(pair[1])
        # we pad all responses into equal length
        processed_batch.outputResponseIDs.append(resp + [PAD_TOKEN] * (max_output_length - len(resp)))
    return processed_batch


def generateBatches(data, batch_size):
    """
    Generate batches of data according to the datasize
    :return:
    """
    random.shuffle(data)
    batches = []
    size = len(data)
    def loadBatches(data, total_size, batch_size_):
        for i in range(0, total_size, batch_size_):
            yield data[i:min(total_size, i + batch_size_)]

    for unprocessed_batch in loadBatches(data, size, batch_size):
        processed_batch = processBatch(unprocessed_batch)
        batches.append(processed_batch)
    return batches


