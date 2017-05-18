import io
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from util import FLAGS
from model import BiRnnAttention
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from tester import test, ensembel_test

# print(tf.__version__)

# Parameters
# =================================================


# 加载词典
vocab_dict = get_vocab_dict()
word_vectors = get_word_vector()
vocab_size = len(vocab_dict)
training_set = LoadTrainData(vocab_dict,
                             data_path="../data/train.csv",
                             query_len_threshold=FLAGS.query_length,
                             seq_len_threshold=FLAGS.sequence_length,
                             batch_size=FLAGS.batch_size)

train_set = LoadTestData(vocab_dict, "../data/train.json", query_len_threshold=FLAGS.query_length,
                         seq_len_threshold=FLAGS.sequence_length, batch_size=-1)
dev_set = LoadTestData(vocab_dict, "../data/dev.json", query_len_threshold=FLAGS.query_length,
                       seq_len_threshold=FLAGS.sequence_length, batch_size=100)
test_set = LoadTestData(vocab_dict, "../data/test.json", query_len_threshold=FLAGS.query_length,
                        seq_len_threshold=FLAGS.sequence_length, batch_size=100)


with tf.variable_scope("MatplotlibInput"):
    img_strbuf_plh = tf.placeholder(tf.string, shape=[])
    my_img = tf.image.decode_png(img_strbuf_plh, 4)
    img_summary = tf.summary.image(
        'DCG@k',
        tf.expand_dims(my_img, 0)
    )


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
with tf.Session(config = config) as sess:
#with tf.Session() as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ", timestamp)
    model_name = "lr{}_bz{}_mg{}_seqlen{}_ebdim{}_{}".format(FLAGS.learning_rate,
                                                             FLAGS.batch_size,
                                                             FLAGS.margin,
                                                             FLAGS.sequence_length,
                                                             FLAGS.embedding_dim,
                                                             timestamp)
    models = []
    model_num = 3
    for i in range(model_num):
        model = BiRnnAttention(sess=sess,
                               q_hidden_dim=FLAGS.query_hidden_unites,
                               a_hidden_dim=FLAGS.answer_hidden_unites,
                               num_layers_1=FLAGS.first_lstm_layers,
                               word_vec_initializer=word_vectors,
                               batch_size=FLAGS.batch_size,
                               vocab_size=vocab_size,
                               embedding_size=FLAGS.embedding_dim,
                               learning_rate=FLAGS.learning_rate,
                               margin=FLAGS.margin,
                               query_len_threshold=FLAGS.query_length,
                               ans_len_threshold=FLAGS.sequence_length,
                               name='model'+str(i),
                               keep_prob=FLAGS.keep_prob)
        models.append(model)


    log_dir = "../logs/" + model_name
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    valid_writer = tf.summary.FileWriter(log_dir + "/valid")
    test_writer = tf.summary.FileWriter(log_dir + "/test")

    init = tf.global_variables_initializer()
    sess.run(init)

    steps = []
    train_DCG_3 = []
    train_DCG_5 = []
    train_DCG_full = []
    val_DCG_3 = []
    val_DCG_5 = []
    val_DCG_full = []
    test_DCG_3 = []
    test_DCG_5 = []
    test_DCG_full = []

    step = 0
    num_epochs = FLAGS.num_epochs
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        for batch_data in training_set.next_batch():
            (_, queries, queries_len, queries_mask), (_, pos_ans, pos_len, pos_mask), (_, neg_ans, neg_len, neg_mask) = batch_data
            for m_idx, model in enumerate(models):
                _, loss, summary, cos12, cos13 = model.train(queries_len, pos_len, neg_len, queries, pos_ans, neg_ans, queries_mask, pos_mask, neg_mask)
            if step % 50 == 0:

                train_set = LoadTestData(vocab_dict, "../data/train.json", query_len_threshold=FLAGS.query_length,
                                         seq_len_threshold=FLAGS.sequence_length, batch_size=-1)
                dev_set = LoadTestData(vocab_dict, "../data/dev.json", query_len_threshold=FLAGS.query_length,
                                       seq_len_threshold=FLAGS.sequence_length, batch_size=100)
                test_set = LoadTestData(vocab_dict, "../data/test.json", query_len_threshold=FLAGS.query_length,
                                        seq_len_threshold=FLAGS.sequence_length, batch_size=100)
                print(step, " - loss:", loss)
                print("cos12:",cos12[:10])
                print("cos13:",cos13[:10])

                print("On training set:\n")

                dcg_3, dcg_5, dcg_full = ensembel_test(sess, models, train_set)
                steps.append(step + 1)
                train_DCG_3.append(dcg_3)
                train_DCG_3.append(dcg_5)
                train_DCG_full.append(dcg_full)

                print("On validation set:\n")
                dcg_3, dcg_5, dcg_full = ensembel_test(sess, models, dev_set)

                val_DCG_3.append(dcg_3)
                val_DCG_5.append(dcg_5)
                val_DCG_full.append(dcg_full)

                print("On test set:\n")
                dcg_3, dcg_5, dcg_full = ensembel_test(sess, models, test_set)

                test_DCG_3.append(dcg_3)
                test_DCG_5.append(dcg_5)
                test_DCG_full.append(dcg_full)
            train_writer.add_summary(summary, step)
            step += 1

        """
        print("On test set:\n")
        dcg_3, dcg_5, dcg_full = test(sess, model, test_set)

        test_DCG_3.append(dcg_3)
        test_DCG_5.append(dcg_5)
        test_DCG_full.append(dcg_full)
        """

        saver = tf.train.Saver(tf.global_variables())

        saver_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"), step)

    x_locator = MultipleLocator(1)  # 将x轴刻度标签设置为1的倍数
    y_locator = MultipleLocator(1)  # 将y轴刻度标签设置为1的倍数

    try:
        print("len val_DCG_3: ", len(val_DCG_3))
        plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(steps, val_DCG_3, "-b", label="DCG@3")
        ax.plot(steps, val_DCG_5, "-r", label="DCG@5")
        ax.plot(steps, val_DCG_full, "-g", label="DCG@full")
        plt.title("Validation Set")
        plt.xlabel("steps")
        plt.ylabel("DCG@k")
        plt.legend(loc="upper left")
        plt.ylim(0, 10)
        ax.yaxis.set_major_locator(y_locator)
        """
        for x in range(len(steps)):
            plt.annotate(round(val_DCG_3[x], 3),
                         xy=(steps[x], val_DCG_3[x]),
                         xytext=(-10, 10),
                         textcoords='offset points')
            plt.annotate(round(val_DCG_5[x], 3),
                         xy=(steps[x], val_DCG_5[x]),
                         xytext=(-10, 10),
                         textcoords='offset points')
            plt.annotate(round(val_DCG_full[x], 3),
                         xy=(steps[x], val_DCG_full[x]),
                         xytext=(-10, 10),
                         textcoords='offset points')
        """
        x = 0
        plt.annotate(round(val_DCG_3[x], 3),
                     xy=(steps[x], val_DCG_3[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(val_DCG_5[x], 3),
                     xy=(steps[x], val_DCG_5[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(val_DCG_full[x], 3),
                     xy=(steps[x], val_DCG_full[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')

        x = steps[-2]
        plt.annotate(round(val_DCG_3[x], 3),
                     xy=(steps[x], val_DCG_3[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(val_DCG_5[x], 3),
                     xy=(steps[x], val_DCG_5[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(val_DCG_full[x], 3),
                     xy=(steps[x], val_DCG_full[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')

        max_val_3 = max(val_DCG_3)
        max_val_5 = max(val_DCG_5)
        max_val_full = max(val_DCG_full)

        for x in range(1, len(steps)-1):
            if val_DCG_3[x] == max_val_3:
                plt.annotate(round(max_val_3, 3),
                             xy=(steps[x], val_DCG_3[x]),
                             xytext=(-10, 10),
                             textcoords='offset points')
            if val_DCG_5[x] == max_val_5:
                plt.annotate(round(max_val_5, 3),
                             xy=(steps[x], val_DCG_5[x]),
                             xytext=(-10, 10),
                             textcoords='offset points')
            if val_DCG_full[x] == max_val_full:
                plt.annotate(round(max_val_full, 3),
                             xy=(steps[x], val_DCG_full[x]),
                             xytext=(-10, 10),
                             textcoords='offset points')

        plt.show()
        imgdata_v = io.BytesIO()
        plt.savefig(imgdata_v, format='png', dpi=200)
        imgdata_v.seek(0)

        plot_img_summary = sess.run(img_summary, feed_dict={img_strbuf_plh: imgdata_v.getvalue()})
        valid_writer.add_summary(plot_img_summary, step + 1)

        plt.clf()
    except Exception as e:
        print(e)

    """
    epochs = range(1, num_epochs + 1)

    plt.figure(1)
    ax = plt.subplot(111)
    ax.plot(epochs, test_DCG_3, "x-b", label="DCG@3")
    ax.plot(epochs, test_DCG_5, "+-r", label="DCG@5")
    ax.plot(epochs, test_DCG_full, "o-g", label="DCG@full")
    plt.title("Test Set")
    plt.xlabel("epochs")
    plt.ylabel("DCG@k")
    plt.legend(loc="upper left")
    plt.ylim(0, 10)
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)
    for x in epochs:
        plt.annotate(round(test_DCG_3[x-1], 3),
                     xy=(x, test_DCG_3[x-1]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(test_DCG_5[x-1], 3),
                     xy=(x, test_DCG_5[x-1]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(test_DCG_full[x-1], 3),
                     xy=(x, test_DCG_full[x-1]),
                     xytext=(-10, 10),
                     textcoords='offset points')
    """
    try:
        print("len test_DCG_3: ", len(test_DCG_3))
        plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(steps, test_DCG_3, "-b", label="DCG@3")
        ax.plot(steps, test_DCG_5, "-r", label="DCG@5")
        ax.plot(steps, test_DCG_full, "-g", label="DCG@full")
        plt.title("Test Set")
        plt.xlabel("epochs")
        plt.ylabel("DCG@k")
        plt.legend(loc="upper left")
        plt.ylim(0, 10)

        ax.yaxis.set_major_locator(y_locator)

        x = 0
        plt.annotate(round(val_DCG_3[x], 3),
                     xy=(steps[x], test_DCG_3[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(val_DCG_5[x], 3),
                     xy=(steps[x], test_DCG_5[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(val_DCG_full[x], 3),
                     xy=(steps[x], test_DCG_full[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')

        x = steps[-2]
        plt.annotate(round(test_DCG_3[x], 3),
                     xy=(steps[x], test_DCG_3[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(test_DCG_5[x], 3),
                     xy=(steps[x], test_DCG_5[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')
        plt.annotate(round(test_DCG_full[x], 3),
                     xy=(steps[x], test_DCG_full[x]),
                     xytext=(-10, 10),
                     textcoords='offset points')

        max_test_3 = max(test_DCG_3)
        max_test_5 = max(test_DCG_5)
        max_test_full = max(test_DCG_full)

        for x in range(1, len(steps)-1):
            if test_DCG_3[x] == max_test_3:
                plt.annotate(round(max_test_3, 3),
                             xy=(steps[x], test_DCG_3[x]),
                             xytext=(-10, 10),
                             textcoords='offset points')
            if test_DCG_5[x] == max_test_5:
                plt.annotate(round(max_test_5, 3),
                             xy=(steps[x], test_DCG_5[x]),
                             xytext=(-10, 10),
                             textcoords='offset points')
            if test_DCG_full[x] == max_test_full:
                plt.annotate(round(max_test_full, 3),
                             xy=(steps[x], test_DCG_full[x]),
                             xytext=(-10, 10),
                             textcoords='offset points')

        plt.show()

        imgdata_t = io.BytesIO()
        plt.savefig(imgdata_t, format='png', dpi=300)
        imgdata_t.seek(0)

        plot_img_summary = sess.run(img_summary, feed_dict={img_strbuf_plh: imgdata_t.getvalue()})
        test_writer.add_summary(plot_img_summary, step + 1)

        plt.clf()
    except Exception as e:
        print(e)

    train_writer.close()
    valid_writer.close()
    test_writer.close()

    with open(log_dir + "/dcg_k_score.txt", "w") as f:
        print("On train set:\n")
        dcg_3, dcg_5, dcg_full = test(sess, model, train_set, number=100)

        print("On validation set:\n")
        v_dcg_3, v_dcg_5, v_dcg_full = test(sess, model, dev_set)

        print("On test set:\n")
        t_dcg_3, t_dcg_5, t_dcg_full = test(sess, model, test_set)

        s1 = "On training set:\n" \
             "DCG@3 Mean: {}\n" \
             "DCG@5 Mean: {}\n" \
             "DCG@full Mean: {}\n" \
             "==============================\n".format(dcg_3, dcg_5, dcg_full)
        s2 = "On validation set:\n" \
             "DCG@3 Mean: {}\n" \
             "DCG@5 Mean: {}\n" \
             "DCG@full Mean: {}\n" \
             "==============================\n".format(v_dcg_3, v_dcg_5, v_dcg_full)
        s3 = "On test set:\n" \
             "DCG@3 Mean: {}\n" \
             "DCG@5 Mean: {}\n" \
             "DCG@full Mean: {}\n" \
             "==============================\n".format(t_dcg_3, t_dcg_5, t_dcg_full)

        s4 = "steps: " + str(steps) + "\n" \
             + "train_DCG_3: " + str(train_DCG_3) + "\n" \
             + "train_DCG_5: " + str(train_DCG_5) + "\n" \
             + "train_DCG_full: " + str(train_DCG_full) + "\n" \
             + "val_DCG_3: " + str(val_DCG_3) + "\n" \
             + "val_DCG_5: " + str(val_DCG_5) + "\n" \
             + "val_DCG_full: " + str(val_DCG_full) + "\n" \
             + "test_DCG_3: " + str(test_DCG_3) + "\n" \
             + "test_DCG_5: " + str(test_DCG_5) + "\n" \
             + "test_DCG_full: " + str(test_DCG_full) + "\n"

        f.write(s1 + s2 + s3 + s4)
