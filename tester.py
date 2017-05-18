import tensorflow as tf
import pandas as pd
import numpy as np
from util import FLAGS


def dcg_k(data, k=3):
    score = 0
    for num in range(k):
        top = np.power(2, data['label'][num]) - 1
        bottom = np.log2(data['rank'][num] + 1)
        score += np.divide(top, bottom)
    return score

def ensembel_test(sess, models, testing_set, filename=None):
    DCG_3 = []
    DCG_5 = []
    DCG_full = []

    zero_3 = 0
    zero_5 = 0
    for batch_data in testing_set.next_batch():
        (test_query_ids, test_queries, test_queries_len, queries_mask), \
         (answers_ids, answers, answers_len, answers_mask, answers_label) = batch_data
        res = np.zeros(shape=[len(answers_label)], dtype=np.float32)
        for model in models:
            tmp_res =np.array( model.predict(test_queries_len, test_queries, answers, answers_len, queries_mask, answers_mask))
            print("res", res.shape, tmp_res.shape)
            res = res+tmp_res

        res = list(zip(test_query_ids, answers_ids, answers_label, res[0].tolist()))

        unique_query_ids = list(set(test_query_ids))
        df = pd.DataFrame(res, columns=['query_id', 'passage_id', 'label', 'score'])

        out_frames = []
        for query_id in unique_query_ids:
            passages = df[df['query_id'] == query_id]
            rank = range(1, passages.count()['label'] + 1)

            # result = passages.sort(['score'], ascending=False).reset_index(drop=True)
            result = passages.sort(['score'], ascending=False).reset_index(drop=True)
            result['rank'] = rank
            result.drop('score', axis=1, inplace=True)
            out_frames.append(result)

            dcg_3 = dcg_k(result, 3)
            dcg_5 = dcg_k(result, 5)
            dcg_full = dcg_k(result, rank[-1])

            if dcg_5 < 0.1:
                zero_3 += 1
                zero_5 += 1
            elif dcg_3 < 0.1:
                zero_3 += 1

            DCG_3.append(dcg_3)
            DCG_5.append(dcg_5)
            DCG_full.append(dcg_full)

        #if filename is not None:
        #    out_df = pd.concat(out_frames)
        #    out_df.to_csv("../result/" + filename, index=False)

    dcg_3_mean = np.mean(DCG_3)
    dcg_5_mean = np.mean(DCG_5)
    dcg_full_mean = np.mean(DCG_full)

    print("number of Zero DCG@3: ", zero_3)
    print("number of Zero DCG@5: ", zero_5)
    print("DCG@3 Mean: ", dcg_3_mean)
    print("DCG@5 Mean: ", dcg_5_mean)
    print("DCG@full Mean: ", dcg_full_mean)
    print("================================")

    return dcg_3_mean, dcg_5_mean, dcg_full_mean

def test(sess, model, testing_set, filename=None):
    DCG_3 = []
    DCG_5 = []
    DCG_full = []

    zero_3 = 0
    zero_5 = 0
    for batch_data in testing_set.next_batch():
        (test_query_ids, test_queries, test_queries_len), \
         (answers_ids, answers, answers_len, answers_label) = batch_data

        res = model.predict(test_queries_len, test_queries, answers, answers_len)

        res = list(zip(test_query_ids, answers_ids, answers_label, res[0].tolist()))

        unique_query_ids = list(set(test_query_ids))
        df = pd.DataFrame(res, columns=['query_id', 'passage_id', 'label', 'score'])

        out_frames = []
        for query_id in unique_query_ids:
            passages = df[df['query_id'] == query_id]
            rank = range(1, passages.count()['label'] + 1)

            # result = passages.sort(['score'], ascending=False).reset_index(drop=True)
            result = passages.sort(['score'], ascending=False).reset_index(drop=True)
            result['rank'] = rank
            result.drop('score', axis=1, inplace=True)
            out_frames.append(result)

            dcg_3 = dcg_k(result, 3)
            dcg_5 = dcg_k(result, 5)
            dcg_full = dcg_k(result, rank[-1])

            if dcg_5 < 0.1:
                zero_3 += 1
                zero_5 += 1
            elif dcg_3 < 0.1:
                zero_3 += 1

            DCG_3.append(dcg_3)
            DCG_5.append(dcg_5)
            DCG_full.append(dcg_full)

        #if filename is not None:
        #    out_df = pd.concat(out_frames)
        #    out_df.to_csv("../result/" + filename, index=False)

    dcg_3_mean = np.mean(DCG_3)
    dcg_5_mean = np.mean(DCG_5)
    dcg_full_mean = np.mean(DCG_full)

    print("number of Zero DCG@3: ", zero_3)
    print("number of Zero DCG@5: ", zero_5)
    print("DCG@3 Mean: ", dcg_3_mean)
    print("DCG@5 Mean: ", dcg_5_mean)
    print("DCG@full Mean: ", dcg_full_mean)
    print("================================")

    return dcg_3_mean, dcg_5_mean, dcg_full_mean

