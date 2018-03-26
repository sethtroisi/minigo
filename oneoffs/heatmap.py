"""
Used to plot a heatmap of the policy and value networks.
Check FLAGS for default values for what models to load.

Usage:
python heatmap.py

"""
import sys; sys.path.insert(0, '.')

import go
import dual_net
import preprocessing
import os
import shipname
import numpy as np
import coords
import tensorflow as tf

from tqdm import tqdm
from gtp_wrapper import make_gtp_instance, MCTSPlayer
from utils import logged_timer as timer
from tensorflow import gfile


tf.app.flags.DEFINE_string(
    "sgf_dir", "sgf/baduk_db/", "sgf database")

tf.app.flags.DEFINE_string("model_dir", "saved_models",
                           "Where the model files are saved")
tf.app.flags.DEFINE_string("data_dir", "data/eval", "Where to save data")
tf.app.flags.DEFINE_integer("idx_start", 150,
                            "Only take models after given idx")
tf.app.flags.DEFINE_integer("eval_every", 5,
                            "Eval every k models to generate the curve")

FLAGS = tf.app.flags.FLAGS

def get_model_paths(model_dir):
    '''Returns all model paths in the model_dir.'''
    all_models = gfile.Glob(os.path.join(model_dir, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = [
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames]
    model_names = sorted(model_numbers_names)
    return [os.path.join(model_dir, name[1]) for name in model_names]


def load_player(model_path):
  print("Loading weights from %s ... " % model_path)
  with timer("Loading weights from %s ... " % model_path):
      network = dual_net.DualNetwork(model_path)
      network.name = os.path.basename(model_path)
  player = MCTSPlayer(network, verbosity=2)
  return player


def restore_params(model_path, player):
  with player.network.sess.graph.as_default():
    player.network.initialize_weights(model_path)


def eval_player(player):
  #TODO(sethtroisi): play at every square and measure value
  pos = go.Position(komi=7.5)
  return player.network.run(pos)


def get_data(model_dir, data_dir, idx_start=100, eval_every=10):
  model_paths = get_model_paths(model_dir)
  player = None

  print("Evaluating models {}-{}, eval_every={}".format(idx_start, len(model_paths), eval_every))
  for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
    if player:
      restore_params(model_paths[idx], player)
    else:
      player = load_player(model_paths[idx])

    probs, value = eval_player(player=player)
    with open(os.path.join(data_dir, "eval-{}".format(idx)), "w") as data:
        data.write("{},  {},  {}\n".format(idx, value, ",".join(map(str, probs))))


def main(unusedargv):
  df = get_data(FLAGS.model_dir, FLAGS.data_dir, idx_start=FLAGS.idx_start, eval_every=FLAGS.eval_every)


FLAGS = tf.app.flags.FLAGS

if __name__ == "__main__":
  tf.app.run(main)
