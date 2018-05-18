import multiprocessing as mp
import time

import tensorflow as tf
from tqdm import tqdm

feature_a = tf.train.Feature(int64_list=tf.train.Int64List(value=range(6000)))
example = tf.train.Example(features=tf.train.Features(feature={
    'ints': feature_a
}))
example = example.SerializeToString()

def get_examples(foo):
    return example
#    return [example]


def collect(items, threads=4):
        writer = tf.python_io.TFRecordWriter("/media/eights/Sojourner/tmp/single")

        count = 0
        count_size = 0
        with mp.Pool(threads) as pool:
#            res = map(get_examples, items)
            res = pool.imap(get_examples, items)
#            for item in tqdm(res, total=len(items)):
#                for proto in item:
            for proto in tqdm(res, total=len(items)):
                    writer.write(proto)

                    count += 1
                    count_size += len(proto)
                    if count % 100000 == 0:
                        print (count, count_size)
#                    if count % 10 == 0:
#                        time.sleep(0.1)


collect(range(10 ** 7))
