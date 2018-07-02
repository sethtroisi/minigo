import argh
import argparse
import datetime as dt
import functools
import itertools
import multiprocessing as mp
import os
import random
import subprocess
import tempfile
import time
from collections import deque

from absl import flags
import tensorflow as tf
from tqdm import tqdm

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

    def make_example(protostring):
        e = tf.train.Example()
        e.ParseFromString(protostring)
        return e
    return list(map(make_example, choices))


def choose(game, samples_per_game=4):
    examples = pick_examples_from_tfrecord(game, samples_per_game)
    timestamp = file_timestamp(game)
    return [(timestamp, ex) for ex in examples]


def file_timestamp(filename):
    try:
        return int(os.path.basename(filename).split('-')[0])
    except:
        return 0


def _ts_to_str(timestamp):
    return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class ExampleBuffer():
    def __init__(self, max_size=2000000, samples_per_game=4):
        self.examples = deque(maxlen=max_size)
        self.max_size = max_size
        self.samples_per_game = samples_per_game
        self.func = functools.partial(
            choose, samples_per_game=self.samples_per_game)

    def parallel_fill(self, games, threads=8):
        """ games is a list of .tfrecord.zz game records. """
        games.sort(key=os.path.basename)
        # A couple extra in case parsing fails
        max_games = self.max_size // self.samples_per_game + 10
        if len(games) > max_games:
            games = games[-max_games:]

        with mp.Pool(threads) as pool:
            res = tqdm(pool.imap(self.func, games), total=len(games))
            self.examples.extend(itertools.chain(*res))

    def update(self, new_games):
        """ new_games is a list of .tfrecord.zz new game records. """
        new_games.sort(key=os.path.basename)
        first_new_game = None
        for idx, game in enumerate(tqdm(new_games)):
            timestamp = file_timestamp(game)
            if timestamp <= self.examples[-1][0]:
                continue
            elif first_new_game is None:
                first_new_game = idx
                print("Found {}/{} new games".format(
                    len(new_games) - idx, len(new_games)))
            self.examples.extend(self.func(game))

    def flush(self, path):
        # random.shuffle on deque is O(n^2) convert to list for O(n)
        self.examples = list(self.examples)
        random.shuffle(self.examples)
        with timer("Writing examples to " + path):
            preprocessing.write_tf_examples(
                path, [ex[1] for ex in self.examples])
        self.examples.clear()
        self.examples = deque(maxlen=self.max_size)

    @property
    def count(self):
        return len(self.examples)

    def __str__(self):
        if self.count == 0:
            return "ExampleBuffer: 0 positions"
        return "ExampleBuffer: {} positions sampled from {} to {}".format(
            self.count,
            _ts_to_str(self.examples[0][0]),
            _ts_to_str(self.examples[-1][0]))


def files_for_model(model):
    return tf.gfile.Glob(os.path.join(LOCAL_DIR, model[1], '*.zz'))


def smart_rsync(
        from_model_num=0,
        source_dir=None,
        dest_dir=LOCAL_DIR):
    source_dir = source_dir or fsdb.selfplay_dir()
    from_model_num = 0 if from_model_num < 0 else from_model_num
    models = [m for m in fsdb.get_models() if m[0] >= from_model_num]
    for _, model in models:
        _rsync_dir(os.path.join(
            source_dir, model), os.path.join(dest_dir, model))


def _rsync_dir(source_dir, dest_dir):
    ensure_dir_exists(dest_dir)
    with open('.rsync_log', 'ab') as rsync_log:
        subprocess.call(['gsutil', '-m', 'rsync', source_dir, dest_dir],
                        stderr=rsync_log)


def fill_and_wait(bufsize=dual_net.EXAMPLES_PER_GENERATION,
                  write_dir=None,
                  model_window=100,
                  threads=8,
                  skip_first_rsync=False):
    """ Fills a ringbuffer with positions from the most recent games, then
    continually rsync's and updates the buffer until a new model is promoted.
    Once it detects a new model, iit then dumps its contents for training to
    immediately begin on the next model.
    """
    write_dir = write_dir or fsdb.golden_chunk_dir()
    buf = ExampleBuffer(bufsize)
    models = fsdb.get_models()[-model_window:]
    # Last model is N.  N+1 is training.  We should gather games for N+2.
    chunk_to_make = os.path.join(write_dir, str(
        models[-1][0] + 2) + '.tfrecord.zz')
    while tf.gfile.Exists(chunk_to_make):
        print("Chunk for next model ({}) already exists.  Sleeping.".format(
            chunk_to_make))
        time.sleep(5 * 60)
        models = fsdb.get_models()[-model_window:]
    print("Making chunk:", chunk_to_make)
    if not skip_first_rsync:
        with timer("Rsync"):
            smart_rsync(models[-1][0] - 6)
    files = tqdm(map(files_for_model, models), total=len(models))
    buf.parallel_fill(list(itertools.chain(*files)), threads=threads)

    print("Filled buffer, watching for new games")
    while fsdb.get_latest_model()[0] == models[-1][0]:
        with timer("Rsync"):
            smart_rsync(models[-1][0] - 2)
        new_files = tqdm(map(files_for_model, models[-2:]), total=len(models))
        buf.update(list(itertools.chain(*new_files)))
        time.sleep(60)
    latest = fsdb.get_latest_model()

    print("New model!", latest[1], "!=", models[-1][1])
    print(buf)
    buf.flush(os.path.join(write_dir, str(latest[0] + 1) + '.tfrecord.zz'))


def make_chunk_for(output_dir=LOCAL_DIR,
                   local_dir=LOCAL_DIR,
                   game_dir=None,
                   model_num=1,
                   positions=dual_net.EXAMPLES_PER_GENERATION,
                   threads=8,
                   samples_per_game=4):
    """
    Explicitly make a golden chunk for a given model `model_num`
    (not necessarily the most recent one).

      While we haven't yet got enough samples (EXAMPLES_PER_GENERATION)
      Add samples from the games of previous model.
    """
    game_dir = game_dir or fsdb.selfplay_dir()
    ensure_dir_exists(output_dir)
    models = [model for model in fsdb.get_models() if model[0] < model_num]
    files = []
    for _, model in sorted(models, reverse=True):
        local_model_dir = os.path.join(local_dir, model)
        if not tf.gfile.Exists(local_model_dir):
            print("Rsyncing", model)
            _rsync_dir(os.path.join(game_dir, model), local_model_dir)
        files.extend(tf.gfile.Glob(os.path.join(local_model_dir, '*.zz')))
        print("{}: {} games".format(model, len(files)))
        if len(files) * samples_per_game > positions:
            break

    output = os.path.join(output_dir, str(model_num) + '.tfrecord.zz')
    _write_chunk(files, output, threads, samples_per_game)


def make_chunk_of_sgf_dir(game_dir,
                          chunk_name,
                          positions=dual_net.EXAMPLES_PER_GENERATION,
                          threads=8,
                          samples_per_game=4):
    assert chunk_name.endswith('.tfrecord.zz'), chunk_name

    if not tf.gfile.Exists(game_dir):
        print("game_dir {} not found".format(game_dir))

    with tempfile.TemporaryDirectory() as record_dir:
        sgfs = []
        records = []
        for root, dirs, files in tf.gfile.Walk(game_dir):
            for name in files:
                if name.endswith('.sgf'):
                    sgfs.append(os.path.join(root, name))
                    record_name = name[:-4] + '.tfrecord.zz'
                    records.append(os.path.join(record_dir, record_name))

        # converting sgfs to tfrecords in the temp directory
        print("Converting sgfs to tfrecords in", record_dir)
        with mp.Pool(threads) as pool:
            records = list(tqdm(
                pool.imap(_process_sgf_to_record, zip(sgfs, records)),
                total=len(sgfs)))

        records = list(filter(None.__ne__, records))
        ensure_dir_exists(os.path.basename(chunk_name))
        _write_chunk(records, chunk_name, positions, threads, samples_per_game)


def _process_sgf_to_record(pair):
    try:
        preprocessing.make_dataset_from_sgf(*pair)
    except Exception as e:
        print("Processing {} failed".format(pair[0]))
        print(e)
        return None
    return pair[1]


def _write_chunk(files, output, positions, threads, samples_per_game):
    print("Filling from {} files".format(len(files)))
    buf = ExampleBuffer(positions, samples_per_game=samples_per_game)
    buf.parallel_fill(files, threads=threads)
    print(buf)
    print("Writing to", output)
    buf.flush(output)


parser = argparse.ArgumentParser()
argh.add_commands(parser, [
    fill_and_wait, smart_rsync, make_chunk_for, make_chunk_of_sgf_dir
])

if __name__ == "__main__":
    import sys
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
