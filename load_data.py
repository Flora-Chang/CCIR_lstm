import numpy as np
import json
from util import FLAGS


def get_vocab_dict(input_file="../data/word_dict.txt"):
    # 返回 {word: id} 字典
    words_dict = {}
    num = 0
    with open(input_file) as f:
        for word in f:
            words_dict[word.strip()] = num
            num += 1

    return words_dict


def get_word_vector(input_file="../data/vectors_word.txt"):
    word_vectors = []
    with open(input_file) as f:
        for line in f:
            line = [float(v) for v in line.strip().split()]
            word_vectors.append(line)
    return word_vectors


def batch(inputs, query_threshold, seq_threshold_length, is_query=False,max_sequence_length=None):
    if is_query:
        threshold_length = query_threshold
    else:
        threshold_length = seq_threshold_length
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    '''
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
        max_sequence_length = min(max_sequence_length, threshold_length)
    '''
    max_sequence_length = threshold_length
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
    mask  = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.bool)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            if j >= threshold_length:
                sequence_lengths[i] = max_sequence_length
                break
            inputs_batch_major[i, j] = element

    for i , seq_len in enumerate(sequence_lengths):
        for j in range(seq_len):
            mask[i,j] = True

    # [batch_size, max_time] -> [max_time, batch_size]
    #inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major, sequence_lengths, mask


class LoadTrainData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, seq_len_threshold, batch_size=64):
        self.vocab_dict = vocab_dict
        self.data_path = data_path
        self.batch_size = batch_size
        self.query_len_thresholf = query_len_threshold
        self.seq_len_threshold = seq_len_threshold  # 句子长度限制
        self.data = open(self.data_path, 'r').readlines()
        self.batch_index = 0
        print("len data: ", len(self.data))

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res

    def next_batch(self, shuffle=True):
        self.batch_index = 0
        data = np.array(self.data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        while self.batch_index < num_batches_per_epoch \
                and (self.batch_index + 1) * self.batch_size <= data_size:
            query_ids = []
            queries = []
            pos_ids = []
            pos_answers = []
            neg_ids = []
            neg_answers = []

            start_index = self.batch_index * self.batch_size
            self.batch_index += 1
            end_index = min(self.batch_index * self.batch_size, data_size)
            batch_data = shuffled_data[start_index:end_index]

            for line in batch_data.tolist():

                line = line.split(',')
                query_id = int(line[0])
                query = list(map(self._word_2_id, line[1].split()))
                pos_id = int(line[2])
                pos_ans = list(map(self._word_2_id, line[3].split()))
                neg_id = int(line[4])
                neg_ans = list(map(self._word_2_id, line[5].split()))

                query_ids.append(query_id)
                queries.append(query)
                pos_ids.append(pos_id)
                pos_answers.append(pos_ans)
                neg_ids.append(neg_id)
                neg_answers.append(neg_ans)

            queries, queries_length, queries_mask = batch(queries, self.query_len_thresholf, self.seq_len_threshold, is_query=True)
            pos_answers, pos_length, pos_mask = batch(pos_answers, self.query_len_thresholf, self.seq_len_threshold)
            neg_answers, neg_length, neg_mask = batch(neg_answers, self.query_len_thresholf, self.seq_len_threshold)

            yield (query_ids, queries, queries_length, queries_mask), \
                    (pos_ids, pos_answers, pos_length, pos_mask), \
                    (neg_ids, neg_answers, neg_length, neg_mask)


class LoadTestData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, seq_len_threshold, batch_size):
        self.vocab_dict = vocab_dict
        self.data_path = data_path
        self.query_len_threshold =query_len_threshold
        self.seq_len_threshold = seq_len_threshold
        self.index = 0
        self.data = open(data_path, 'r').readlines()
        self.batch_size = batch_size
        self.data_size = len(self.data)

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res


    def next_batch(self):
        if self.batch_size == -1:
            self.batch_size = 100
            self.data_size = self.batch_size
        while(self.index + 1) * self.batch_size <= self.data_size:
            batch_data = self.data[self.index * self.batch_size: (self.index +1) * self.batch_size]
            self.index += 1
            queries = []
            query_ids = []
            answers = []
            answers_ids = []
            answers_label = []

            for line in batch_data:
                line = json.loads(line)
                passages = line['passages']
                query_id = line['query_id']
                query = list(map(self._word_2_id, line['query']))
                for passage in passages:
                    passage_id = passage['passage_id']
                    label = passage['label']
                    passage_text_list = list(map(self._word_2_id, passage['passage_text'].split()))
                    queries.append(query)
                    query_ids.append(query_id)
                    answers_ids.append(passage_id)
                    answers_label.append(label)
                    answers.append(passage_text_list)

            queries, queries_len, queries_mask = batch(queries, self.query_len_threshold, self.seq_len_threshold, is_query=True)
            answers, answers_len, answers_mask = batch(answers, self.query_len_threshold, self.seq_len_threshold)
            yield (query_ids, queries, queries_len, queries_mask), (answers_ids, answers, answers_len, answers_mask, answers_label)
