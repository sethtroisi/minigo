import itertools
import multiprocessing as mp
import time

import tensorflow as tf
from tqdm import tqdm

temp = list(range(6000))
feature_a = tf.train.Feature(int64_list=tf.train.Int64List(value=temp))
example = tf.train.Example(features=tf.train.Features(feature={
    'ints': feature_a
}))
example = example.SerializeToString()

def get_examples(foo):
    return example


def collect(items, threads=4):
        writer = tf.python_io.TFRecordWriter("/media/eights/Sojourner/tmp/single")

        count = 0
        count_size = 0
        with mp.Pool(threads, maxtasksperchild=10) as pool:
#            res = map(get_examples, items)
            res = pool.imap(get_examples, items, chunksize=10)
            for proto in tqdm(res):
                proto = proto[:]
                writer.write(proto)

                count += 1
                count_size += len(proto)
                if count % 100000 == 0:
                    print (count, count_size)
#                if count % 10 == 0:
#                    time.sleep(0.1)


collect(itertools.repeat(6000, 10 ** 6))
