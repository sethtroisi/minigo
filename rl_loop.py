# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper scripts to ensure that main.py commands are called correctly."""
import argparse
import logging
import os
import sys
import time

from absl import flags
import argh
from tensorflow import gfile

import cloud_logging
import fsdb
import main
import shipname

# How many games before the selfplay workers will stop trying to play more.
MAX_GAMES_PER_GENERATION = 10000

# How many games minimum, until the trainer will train
MIN_GAMES_PER_GENERATION = 500

# What percent of games to holdout from training per generation
HOLDOUT_PCT = 0.05


def bootstrap(working_dir):
    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join(fsdb.models_dir(), bootstrap_name)
    print("Bootstrapping with working dir {}\n Model 0 exported to {}".format(
        working_dir, bootstrap_model_path))
    main.bootstrap(working_dir, bootstrap_model_path)


def selfplay(verbose=2):
    _, model_name = fsdb.get_latest_model()
    games = gfile.Glob(os.path.join(fsdb.selfplay_dir(), model_name, '*.zz'))
    if len(games) > MAX_GAMES_PER_GENERATION:
        print("{} has enough games ({})".format(model_name, len(games)))
        time.sleep(10*60)
        sys.exit(1)
    print("Playing a game with model {}".format(model_name))
    model_save_path = os.path.join(fsdb.models_dir(), model_name)
    game_output_dir = os.path.join(fsdb.selfplay_dir(), model_name)
    game_holdout_dir = os.path.join(fsdb.holdout_dir(), model_name)
    sgf_dir = os.path.join(fsdb.sgf_dir(), model_name)
    main.selfplay(
        load_file=model_save_path,
        output_dir=game_output_dir,
        holdout_dir=game_holdout_dir,
        output_sgf=sgf_dir,
        holdout_pct=HOLDOUT_PCT,
        verbose=verbose,
    )


def train(working_dir):
    model_num, model_name = fsdb.get_latest_model()

    games = gfile.Glob(os.path.join(fsdb.selfplay_dir(), model_name, '*.zz'))
    if len(games) < MIN_GAMES_PER_GENERATION:
        print("{} doesn't have enough games to train a new model yet ({})".format(
            model_name, len(games)))
        print("Sleeping...")
        time.sleep(10*60)
        print("Done...")
        sys.exit(1)

    print("Training on gathered game data, initializing from {}".format(model_name))
    new_model_num = model_num + 1
    new_model_name = shipname.generate(new_model_num)
    print("New model will be {}".format(new_model_name))
    training_file = os.path.join(
        fsdb.golden_chunk_dir(), str(new_model_num) + '.tfrecord.zz')
    while not gfile.Exists(training_file):
        print("Waiting for", training_file)
        time.sleep(1*60)
    print("Using Golden File:", training_file)

    local_copy = os.path.join(working_dir, str(new_model_num) + '.tfrecord.zz')
    print("Copying locally to ", local_copy)
    gfile.Copy(training_file, local_copy, overwrite=True)

    try:
        save_file = os.path.join(fsdb.models_dir(), new_model_name)
        main.train(working_dir, [local_copy], save_file)
    except:
        logging.exception("Train error")
    finally:
        os.remove(local_copy)


def validate(working_dir, model_num=None, validate_name=None):
    """ Runs validate on the directories up to the most recent model, or up to
    (but not including) the model specified by `model_num`
    """
    if model_num is None:
        model_num, model_name = fsdb.get_latest_model()
    else:
        model_num = int(model_num)
        model_name = fsdb.get_model(model_num)

    # Model N was trained on games up through model N-2, so the validation set
    # should only be for models through N-2 as well, thus the (model_num - 1)
    # term.
    models = list(
        filter(lambda num_name: num_name[0] < (model_num - 1), fsdb.get_models()))
    # Run on the most recent 50 generations,
    # TODO(brianklee): make this hyperparameter dependency explicit/not hardcoded
    holdout_dirs = [os.path.join(fsdb.holdout_dir(), pair[1])
                    for pair in models[-50:]]

    main.validate(working_dir, *holdout_dirs,
                  checkpoint_name=os.path.join(fsdb.models_dir(), model_name),
                  validate_name=validate_name)


def backfill():
    models = [m[1] for m in fsdb.get_models()]

    import dual_net
    import tensorflow as tf
    from tqdm import tqdm
    features, labels = dual_net.get_inference_input()
    dual_net.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                      dual_net.get_default_hyperparams())

    for model_name in tqdm(models):
        if model_name.endswith('-upgrade'):
            continue
        try:
            load_file = os.path.join(fsdb.models_dir(), model_name)
            dest_file = os.path.join(fsdb.models_dir(), model_name)
            main.convert(load_file, dest_file)
        except:
            print('failed on', model_name)
            continue


parser = argparse.ArgumentParser()

argh.add_commands(parser, [train, selfplay, backfill,
                           bootstrap, fsdb.game_counts, validate])

if __name__ == '__main__':
    cloud_logging.configure()
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
