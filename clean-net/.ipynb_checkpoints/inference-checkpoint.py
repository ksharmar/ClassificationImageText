# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Inference

Make predictions on a list of data
Sample command for running validation once:
    python inference.py \
        --data_dir=${DATA_DIR} \
        --class_list=${CLASS_LIST_FILE} \
        --output_file=${OUTPUT_FILE} \
        --checkpoint_dir=${CHECKPOINT_DIR} \
        --mode=val \
        --val_sim_thres=0.2

Sample command for making predictions on a image feature list without verification labels:
    python inference.py \
        --data_dir=${DATA_DIR} \
        --image_feature_list=${IMAGE_FEATURE_LIST_FILE} \
        --class_list=${CLASS_LIST_FILE} \
        --output_file=${OUTPUT_FILE} \
        --checkpoint_dir=${CHECKPOINT_DIR} \
        --mode=inference \
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import shutil
import math
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import tensorflow as tf

import data_provider_factory
from model import CleanNet
from metrics import similarity, accuracy
from losses import total_loss


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', './data/',
                           'Data directory containing all.npy.')
tf.app.flags.DEFINE_string('image_feature_list', './data/input_tsv.tsv',
                           'Image feature list tsv, columns are: [key, url, class_name, feature] or [key, class_name, feature]')
tf.app.flags.DEFINE_string('class_list', './data/classes.txt',
                           'List of class name.')
tf.app.flags.DEFINE_string('checkpoint_dir', './runs/',
                           'Directory to keep checkpoints and logs.')
tf.app.flags.DEFINE_string('output_file', './runs/pred.txt',
                           'File path to output prediction.')
tf.app.flags.DEFINE_integer('img_dim', 2048,
                            'Dimensionality of the image embedding')
tf.app.flags.DEFINE_integer('num_ref', 32,
                            'Number of reference image embeddings for a class')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Batch size.')
tf.app.flags.DEFINE_float('val_sim_thres', 0.1, 
                          'Similarity threshold for validation.')
tf.app.flags.DEFINE_string('embed_norm', 'log',
                           'log|l2_norm|no_norm')
tf.app.flags.DEFINE_string('mode', 'inference',
                           'inference|val')
tf.app.flags.DEFINE_integer('num_samples', -1,
                            'Number of samples to infer. -1 indicates evaluating all data')
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    opt = FLAGS
    
    if opt.mode == "avgbaseline":
        tf.logging.info("Start avg baseline validate once...")
        validate_once(None, None, None, opt)
    else:
        tf.logging.info("Build CleanNet...")
        model = CleanNet(opt.num_ref, opt.img_dim, opt.embed_norm, dropout_rate=0.0, weight_decay=0.0)

        # phi_s: class embedding (batch_size, embed_size)
        # v_q: query image feature (batch_size, img_dim)
        # phi_q: query embedding (batch_size, embed_size)
        # v_qr: reconstructed query image feature (batch_size, img_dim)
        phi_s, v_q, phi_q, v_qr = model.forward(is_training=True)
        cos_sim = similarity(phi_s, phi_q)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # check for checkpoint
        model_path = tf.train.latest_checkpoint(opt.checkpoint_dir, latest_filename=None)
        if not model_path:
            tf.logging.info("Skipping evaluation. No checkpoint found in: %s", opt.checkpoint_dir)
            return

        with tf.Session() as sess:
            # Load model from checkpoint.
            tf.logging.info("Loading model from checkpoint: %s", model_path)
            saver.restore(sess, model_path)
            global_step = tf.train.global_step(sess, model.global_step)
            tf.logging.info("Successfully loaded %s at global step = %d.", \
                            os.path.basename(model_path), global_step)

            if opt.mode == "inference":
                tf.logging.info("Start inference...")
                inference(sess, model, cos_sim, opt)  
            else:
                tf.logging.info("Start validate once...")
                validate_once(sess, model, cos_sim, opt)


def inference(sess, model, cos_sim, opt):
    """Inference"""

    # get data loader
    tf.logging.info("Get data batcher...")
    infer_data_batcher = data_provider_factory.get_data_batcher('inference', 'inference', opt)

    with open(opt.output_file, 'w') as fout:
        count = 0
        while True:
            batch_data = infer_data_batcher.get_batch(opt.batch_size)
            if not batch_data or (opt.num_samples != -1 and count >= opt.num_samples):
                break
            batch_class_id, batch_q, batch_ref = batch_data
            cos_sim_result = sess.run([cos_sim], feed_dict={model.reference: batch_ref, model.query: batch_q})
            for sim in cos_sim_result: 
                fout.write("{}\n".format(sim))
                count += 1
                if count >= opt.num_samples:
                    break
            sys.stdout.write('\r>> Predict %d samples.' % (count))
        sys.stdout.write('\n')
        sys.stdout.flush()


# def validate_once(sess, model, cos_sim, opt):
#     """Run validation once"""

#     # get data loader
#     tf.logging.info("Get data batcher...")
#     val_data_batcher = data_provider_factory.get_data_batcher('trainval', 'val', opt)

#     with open(opt.output_file, 'w') as fout:
#         cumulative_samples = 0.
#         cumulative_correct_pred = 0.
#         steps = int(val_data_batcher.data_size/opt.batch_size)
#         for i in range(steps):
#             _, batch_vlabel, batch_q, batch_vflag, batch_ref = val_data_batcher.get_batch(opt.batch_size)
#             cos_sim_result = sess.run(cos_sim, feed_dict={model.reference: batch_ref, model.query: batch_q})
#             # for j in range(opt.batch_size): (KS)
#             for j in range(batch_q.shape[0]):
#                 fout.write("{}\n".format(cos_sim_result[j]))
#                 cumulative_samples += 1
#                 if (batch_vlabel[j] == 1 and cos_sim_result[j] >= opt.val_sim_thres) or \
#                    (batch_vlabel[j] == 0 and cos_sim_result[j] < opt.val_sim_thres):
#                     cumulative_correct_pred += 1
#             sys.stdout.write('\r>> Predict for %d batches.' % (cumulative_samples))
#         sys.stdout.write('\n')
#         sys.stdout.flush()

#         avg_acc = float(cumulative_correct_pred)/cumulative_samples
#         tf.logging.info('acc = {}'.format(avg_acc))

def validate_once(sess, model, cos_sim, opt):
    """Run validation once"""

    # get data loader
    tf.logging.info("Get data batcher...")
    val_data_batcher = data_provider_factory.get_data_batcher('trainval', 'val', opt)

    with open(opt.output_file, 'w') as fout:
        
        preds = []
        targets = []
        
        cumulative_samples = 0.
        cumulative_correct_pred = 0.
        tp = 0
        total_noise = 0
        total_prec_denom = 0
        class_correct_pred = defaultdict(int)
        class_total = defaultdict(int)
        
        steps = int(val_data_batcher.data_size/opt.batch_size)
        for i in range(steps):
            batch_clsid, batch_vlabel, batch_q, batch_vflag, batch_ref = val_data_batcher.get_batch(opt.batch_size)
            
            if sess is not None:
                cos_sim_result = sess.run(cos_sim, feed_dict={model.reference: batch_ref, model.query: batch_q})
            else:
                # avg baseline
                ref_avg_vec = np.mean(batch_ref, axis=1)
                cos_sim_result = np.diag(cosine_similarity(batch_q, ref_avg_vec))
#                 np.zeros(batch_q.shape[0])
#                 for j in range(batch_q.shape[0]):
#                     query_vec = batch_q[j]
#                     ref_avg_vec = np.mean(batch_ref[j], axis=0)
#                     cos_sim_result[j] = cosine_similarity(query_vec, ref_avg_vec)  
                    # normalization
#                     print("q", ",".join(map(str, query_vec[0:10])))
#                     print('r', ",".join(map(str, ref_avg_vec[0:10])))
#                     sys.exit()
#                     if opt.embed_norm == 'log':
#                         v_q = np.log(query_vec + 1.0)
#                         v_S = np.log(batch_ref[j] + 1.0)
#                     # elif self.embed_norm == 'l2norm':
#                     #    v_q = tf.nn.l2_normalize(self.query, 1)
#                     #    v_S = tf.nn.l2_normalize(self.reference, 2)
#                     elif opt.embed_norm == 'no_norm':
#                         v_q = query_vec
#                         v_S = batch_ref[j]
#                     else:
#                         raise NotImplementedError
                        
#                     v_q = v_q / (np.linalg.norm(v_q) + 1e-16)
#                     v_S = v_S / (np.linalg.norm(v_S) + 1e-16)
#                     cos_sim_result[j] = np.sum(np.multiply(v_q, v_S)) # tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(x1,1),tf.nn.l2_normalize(x2,1)), axis=1)
            
            # for j in range(opt.batch_size): (KS)
            for j in range(batch_q.shape[0]):
                fout.write("{}\n".format(cos_sim_result[j]))
                
                cumulative_samples += 1
                class_total[int(batch_clsid[j])] += 1 
                
                if cos_sim_result[j] < opt.val_sim_thres:
                    pred_j = 0 # noise
                else:
                    pred_j = 1 # clean
                preds.append(pred_j)
                true_j = batch_vlabel[j]  # 1 = clean and 0 = noise
                targets.append(true_j)
                
                if pred_j == true_j:
                    cumulative_correct_pred += 1
                    class_correct_pred[int(batch_clsid[j])] += 1
                
                if true_j == 0 and pred_j == 0:
                    tp += 1
                if true_j == 0:
                    total_noise += 1
                if pred_j == 0:
                    total_prec_denom += 1
                
                    
            sys.stdout.write('\r>> Predict for %d batches.' % (cumulative_samples))
        sys.stdout.write('\n')
        sys.stdout.flush()

        avg_acc = float(cumulative_correct_pred)/cumulative_samples
        tf.logging.info('acc = {} err = {}'.format(avg_acc, 1.0 - avg_acc))
        
        if total_noise == 0:
            tf.logging.info('recall = nan since total_noise=0')
            recall = None
        else:
            recall = float(tp) / total_noise
            tf.logging.info('recall = {}'.format(recall))
        
        if total_prec_denom == 0:
            tf.logging.info('precision = nan since total_prec_denom=0')
            precision = None
        else:
            precision = float(tp) / total_prec_denom
            tf.logging.info('precision = {}'.format(precision))

        if recall and precision:
            f1_noise = 2 * precision * recall / (precision + recall)
            tf.logging.info('f1_noise = {}'.format(f1_noise))
        
        # avg error rate per class.
        nb_cls = len(class_total)
        acc_cls = np.zeros(nb_cls)
        for i in range(nb_cls):
            acc_cls[i] = float(class_correct_pred[i]) / class_total[i]
        avg_cls_acc = np.mean(acc_cls)
        tf.logging.info('avg acc (cls) = {} avg error (cls) = {}'.format(avg_cls_acc, 1.0 - avg_cls_acc))
        
        print('f1_metrics (clean)', f1_score(targets, preds, average='binary'))
        print('f1_metrics (noise)', f1_score(targets, preds, pos_label=0, average='binary'))
        print('f1_metrics (accuracy/micro)', f1_score(targets, preds, average='micro'))
        print('f1_metrics (macro/unweighted mean)', f1_score(targets, preds, average='macro'))
        print('f1_metrics (weighted mean of f1)', f1_score(targets, preds, average='weighted'))

if __name__ == '__main__':
    tf.app.run()
