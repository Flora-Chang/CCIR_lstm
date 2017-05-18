import tensorflow as tf
flags = tf.app.flags

# Model parameters
flags.DEFINE_integer("query_hidden_unites", 50, "query lstm hidden unites numbers")
flags.DEFINE_integer("answer_hidden_unites", 50, "answers lstm hidden unites numbers")
flags.DEFINE_integer("embedding_dim", 50, "words embedding size")
flags.DEFINE_integer("first_lstm_layers", 1, "layers of first lstm layer")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep prob")

# Training / test parameters
flags.DEFINE_integer("sequence_length", 100, "threshold value of sequence length")
flags.DEFINE_integer("query_length", 10, "threshold value of sequence length")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_epochs", 3, "number of epochs")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
flags.DEFINE_float("margin", 1, "cos margin")
flags.DEFINE_bool("c2q_att", True, "context_2_query attention?")
flags.DEFINE_bool("q2c_att", True, "query_2_context attention?")

flags.DEFINE_float("wd",0.0,  " L2 weight decay for regularization?")

FLAGS = flags.FLAGS
