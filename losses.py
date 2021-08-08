import numpy as np
import sys
import tensorflow as tf



def dice_multi(y_actual, y_predicted):
    flat_logits = tf.reshape(y_predicted, [-1, NUM_CLASSES])
    flat_labels = tf.reshape(y_actual, [-1, NUM_CLASSES])
    
    dice_multi = 0
    for index in range(NUM_CLASSES):
        flat_logits_ = tf.nn.softmax(flat_logits)[:, index]
        flat_labels_ = flat_labels[:, index]

        inse = tf.reduce_sum(flat_logits_ * flat_labels_)
        l = tf.reduce_sum(flat_logits_ * flat_logits_)
        r = tf.reduce_sum(flat_labels_ * flat_labels_)
        dice = 2 * (inse) / (l + r)
        dice = tf.clip_by_value(dice, 0, 1 - 1e-10)

        dice_multi += dice

    loss = NUM_CLASSES * 1.0 - dice_multi
    return loss




def youden(target, input):
    lamb = 0.1*np.ones((NUM_CLASSES,NUM_CLASSES))
    for i in range(NUM_CLASSES):
        lamb[i,i]=1
    power = 1 

    eps=0.00000001
    flat_input = tf.reshape(input, [-1, NUM_CLASSES])
    ymask = tf.reshape(target, [-1, NUM_CLASSES])
    log_p=tf.nn.log_softmax(flat_input,axis=1)

    p=tf.nn.softmax(flat_input,axis=1)
    lossd=0

    all_sums = tf.math.reduce_sum(ymask,axis=0)

    final_mat = tf.matmul(tf.transpose(p),ymask)

    for i in range(NUM_CLASSES):
        ni = all_sums[i]
        for j in range(NUM_CLASSES):
            if i==j:
                continue
            nj = all_sums[j]
            tmp_loss = tf.cond(tf.greater(ni*nj, 0), lambda: -lamb[i][j] * tf.math.log( (0.5*(final_mat[i,i]/ni -final_mat[i,j]/nj)+0.5)**power  + eps ), lambda: tf.constant(0,tf.float32))
            lossd+= tmp_loss

    return lossd




def cross_entropy(y_actual, y_predicted):
    flat_logits = tf.reshape(y_predicted, [-1, NUM_CLASSES])
    flat_labels = tf.reshape(y_actual, [-1, NUM_CLASSES])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))
    return loss




    