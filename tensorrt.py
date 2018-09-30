#!/usr/bin/env python3
import time

import numpy as np

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from tensorflow.core.framework import graph_pb2

import sgf_wrapper
import features as features_lib
import dual_net


def load_graph_def(f):
    with open(f, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def save_graph_def(f, graph_def):
    with open(f, 'wb') as f:
        f.write(graph_def.SerializeToString())

model_f = 'saved_models/000721-eagle.pb'
test_sgf = 'data/test.sgf'
batch_size = 4
outputs = ['value_output', 'policy_output']

positions = [n.position for n in sgf_wrapper.replay_sgf_file(test_sgf)]
processed = list(map(features_lib.extract_features, positions))
chunks = [processed[batch_size*i:batch_size*(i+1)]
            for i in range(len(processed) // batch_size)]

processed = np.array(processed).astype(np.float32)
print ("Shape", processed.shape)

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.5))

#####

graph_def = load_graph_def(model_f)

trt_graph = trt.create_inference_graph(
    graph_def,
    outputs,
    max_batch_size=batch_size,
    max_workspace_size_bytes=1600 * 2 ** 10,
    precision_mode='FP32')

# Save new graph
save_graph_def(model_f.replace('.pb', '_trt.pb'), trt_graph)

print()
print(len(processed), 'positions in test sgf')
with tf.Graph().as_default():
    print('Starting')

    dataset = tf.data.Dataset.from_tensor_slices(processed).batch(batch_size, True)
    iterator = dataset.repeat().make_one_shot_iterator()

    placeholder = tf.placeholder(tf.float32,
        [batch_size, 19, 19, features_lib.NEW_FEATURES_PLANES],
        name='pos_tensor')

    return_elements = tf.import_graph_def(
        graph_def=trt_graph,
#        input_map={'pos_tensor': iterator.get_next()},
        input_map={'pos_tensor': placeholder},
        return_elements=outputs)

    outputs = [op.outputs[0] for op in return_elements]
    print('Output:', outputs)

    with tf.Session(config=config) as sess:
        tf.logging.info("Starting inferences.")
        timings = []
        for i in range(110):
            chunk = chunks[i % len(chunks)]

            start = time.time()
#            value = sess.run(outputs)
            value = sess.run(outputs, feed_dict={placeholder: chunk})
            print ('Value:', value[0], len(value[1]))
            timings.append(time.time() - start)

    timings = np.array(timings)
    print(sum(timings), timings)

print("Batch size:", batch_size)
print("Avg: {:.2f}".format(np.average(timings)))
warmed = timings[10:]
avg = np.average(warmed)
print("Avg after warm-up: {:.4f}".format(avg))
print("NPS              : {:.2f}".format(batch_size / avg))
print("Std dev (after w): {:.4f}".format(np.std(warmed)))
