import tensorflow as tf
from tensorflow.python.util import nest
from functools import reduce
from operator import mul
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

VERY_NEGATIVE_NUMBER = -1e30

# convert a tensor like [n,jx,jq,d] to [n*jx*jq, d]
def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat
# convert tensor like [n*jx*jq, d] to [n, jx, jq, d]
def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out
def add_wd(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    with tf.name_scope("weight_decay"):
        for var in variables:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="{}/wd".format(var.op.name))
            tf.add_to_collection("losses", weight_decay)

def exp_mask(val, mask, name=None):
    '''Give very negative number to unmasked elements in val
    For example: [-3, -2, 10], [True, True False] -> [-3, -2, -1e9]'''
    if name is None:
        name = "exp_mask"
    print("type")
    print(val, mask)
    return tf.add(val, (1.0 - tf.cast(mask, dtype=tf.float32)) * VERY_NEGATIVE_NUMBER, name=name)

# a linear function
# map sum_i(args[i]*w[i]) , where w[i] is a variable
def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=True, wd=0.0, input_keep_prob=1.0, is_train=True):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("args must be specified")
    if not nest.is_sequence(args):
        args = [args]
    flat_args = [flatten(arg, 1) for arg in args]
    is_train = tf.convert_to_tensor(is_train, dtype=tf.bool)
    #if input_keep_prob is not None:
        #assert is_train is not None
    flat_args = [tf.cond(is_train, lambda : tf.nn.dropout(arg, input_keep_prob), lambda : arg) for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    out = reconstruct(flat_out, args[0], 1)
    shape = out.get_shape().as_list()
    shape.pop()
    print("shape", shape)

    if squeeze:
        #out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
        out = tf.squeeze(out, axis=[3])

    #out = tf.reshape(out, shape=shape)

    if wd:
        add_wd(wd)
    print("out", out)
    return out

def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=True):
    with tf.variable_scope(scope or "linear_logits"):
        logits =linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope="first",
                       wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        print("logits", logits)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits
# compute the shared matrix
def get_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=True, func=None):
    #call the function of a = w*[args[0], args[1], args[0]*args[1]
    print("args:")
    assert len(args) == 2
    new_arg = args[0]*args[1]
    print(args[0], args[1], new_arg)
    ans = linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                  is_train=is_train)
    return ans
#do softmax on the shared matrix ->logits
def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)
        return out
# target:[?, FLAGS.sequence_length, FLAGS.query_length, hidden_units]
#logits:[?,FLAGS.sequence_length, FLAGS.query_length]
#return: [?,FLAGS.sequence_length, hidden_units]
def softsel(target, logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank-2)
        return out