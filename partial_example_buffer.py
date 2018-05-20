import argh
import argparse
import datetime as dt
import functools
import itertools
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from collections import deque

from tqdm import tqdm
from absl import flags
import tensorflow as tf

import preprocessing
import dual_net
from utils import timer, ensure_dir_exists
import fsdb


READ_OPTS = preprocessing.TF_RECORD_CONFIG

LOCAL_DIR = "data/"

def pick_examples_from_tfrecord(filename, samples_per_game=4):
    protos = list(tf.python_io.tf_record_iterator(filename, READ_OPTS))
    if len(protos) < 20:  # Filter games with less than 20 moves
        return []
    choices = random.sample(protos, min(len(protos), samples_per_game))
    return choices


class BigShuffle():
    def shuffle_and_collect(path, games, threads=4, samples_per_game=4, num_shards=10):
        # 1. Determine roughly how many shards
        # 2. Write (buffered) each proto to one of the shards
        # 3. Randomize each shard
        # 4. Join each shard into one file

        # Avoid Step 1 by allocating 1Gig of memory
        #shard_buffer_size = 10 ** 9 // num_shards

        shard_names = []
        shards = []
        for i in range(num_shards):
            shard_name = "shuffle.shard-{}-of-{}".format(i, num_shards)
            shard_path = os.path.join("/media/eights/Sojourner/tmp", shard_name)
            shard_names.append(shard_path)

            # Writes raw data
            writer = tf.python_io.TFRecordWriter(shard_path)
            shards.append(writer)

        func = functools.partial(
            pick_examples_from_tfrecord,
            samples_per_game=samples_per_game)

        #'''
        count = 0
        count_size = 0
        with timer("Shuffling to shards"), mp.Pool(threads, maxtasksperchild=1) as pool:
            res = map(func, games)
#            res = pool.imap(func, games, chunksize=100)
            for game in tqdm(res, total=len(games)):
                for proto in game:
                    count += 1
                    count_size += len(proto)
                    # Step 2, Step 3: ideally these writes are buffered
                    shard = random.choice(shards)
                    shard.write(proto)
                    #shard.flush() # Test if this has an impact on memory

                    if count % 100000 == 0:
                        print (count, count_size)

        print (count, "positions processed", count_size)
        for shard in shards:
            shard.flush()
            shard.close()
        #'''


        #'''
        joined_writer = tf.python_io.TFRecordWriter(path, READ_OPTS)
        with timer("Shuffling and joining shards"), mp.Pool(threads) as pool:
            for shard_name in tqdm(shard_names):
                res = list(tf.python_io.tf_record_iterator(shard_name))
                random.shuffle(res)
                for proto in tqdm(res):
                    joined_writer.write(proto)
        joined_writer.flush()
        joined_writer.close()
        #'''



def make_chunk_for(
      output_dir=LOCAL_DIR, local_dir=LOCAL_DIR,
      chunk_name='chunk', threads=8, samples_per_game=4, num_shards=10):
    """
    Explicitly make a golden chunk
    """
    ensure_dir_exists(output_dir)
    files = []
    for filename in os.listdir(local_dir):
        if filename.endswith('.tfrecord.zz'):
            path = os.path.join(local_dir, filename)
            files.append(path)

    print ("Filling from {} files".format(len(files)))
    BigShuffle.shuffle_and_collect(
            chunk_name + '.tfrecord.zz',
            files,
            threads=threads,
            samples_per_game=samples_per_game,
            num_shards=num_shards)


parser = argparse.ArgumentParser()
#argh.add_commands(parser, [fill_and_wait, smart_rsync, make_chunk_for])
argh.add_commands(parser, [make_chunk_for])

if __name__ == "__main__":
    import sys
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
