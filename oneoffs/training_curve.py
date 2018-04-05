"""
Used to plot the accuracy of the policy and value networks in
predicting professional game moves and results over the course
of training. Check FLAGS for default values for what models to
load and what sgf files to parse.

Usage:
python training_curve.py

Sample 3 positions from each game
python training_curve.py --num_positions=3

Only grab games after 2005 (default is 2000)
python training_curve.py --min_year=2005
"""
import sys; sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')

import oneoff_utils

import go
import dual_net
import sgf_wrapper
import utils
import os
import shipname
import main
import numpy as np
import itertools
import matplotlib.pyplot as plt
import coords
import pandas as pd
import tensorflow as tf
import sgf
import pickle

from tqdm import tqdm
from sgf_wrapper import sgf_prop
from gtp_wrapper import make_gtp_instance, MCTSPlayer
from utils import logged_timer as timer
from tensorflow import gfile

tf.app.flags.DEFINE_string(
    "sgf_dir", "sgf/baduk_db/", "sgf database")

tf.app.flags.DEFINE_string("model_dir", "saved_models",
                           "Where the model files are saved")
tf.app.flags.DEFINE_string("plot_dir", "data", "Where to save the plots.")
tf.app.flags.DEFINE_integer("min_year", "2000",
                            "Only take sgf games with date >= min_year")
tf.app.flags.DEFINE_string("komi", "7.5",
                            "Only take sgf games with given komi")
tf.app.flags.DEFINE_integer("idx_start", 150,
                            "Only take models after given idx")
tf.app.flags.DEFINE_integer("num_positions", 1,
                            "How many positions from each game to sample from.")
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


def batch_run_many(player, positions, batch_size=100):
  """Used to avoid a memory oveflow issue when running the network
  on too many positions. TODO: This should be a member function of
  player.network?"""
  prob_list = []
  value_list = []
  for idx in range(0, len(positions), batch_size):
    probs, values = player.network.run_many(positions[idx:idx+batch_size])
    prob_list.append(probs)
    value_list.append(values)
  return np.concatenate(prob_list, axis=0), np.concatenate(value_list, axis=0)


def eval_player(temp_values, player, positions, moves, results):
  probs, values = batch_run_many(player, positions)
  policy_moves = [coords.from_flat(c) for c in np.argmax(probs, axis=1)]
  temp_values.append((policy_moves, values.tolist()))
  top_move_agree = [moves[idx] == policy_moves[idx] for idx in range(len(moves))]
  square_err = (values - results)**2/4
  return top_move_agree, square_err


def sample_positions_from_games(rand, sgf_files, num_positions=1):
  pos_data = []
  move_data = []
  result_data = []
  move_idxs = []

  fail_count = 0
  for i, path in tqdm(enumerate(sgf_files)):
    if i % 1000 == 0:
      print(i)
    try:
      positions, moves, results = oneoff_utils.parse_sgf(path)
    except KeyboardInterrupt:
      raise
    except Exception as e:
      fail_count += 1
      print("Fail {}, while parsing {}: {}".format(fail_count, path, e))
      continue

    #add entire game
    if num_positions== -1:
      pos_data.extend(positions)
      move_data.extend(moves)
      move_idxs.extend(range(len(positions)))
      result_data.extend(results)
    else:
      for idx in rand.choice(len(positions), num_positions):
        pos_data.append(positions[idx])
        move_data.append(moves[idx])
        result_data.append(results[idx])
        move_idxs.append(idx)
  print("Sampled {} positions, failed to parse {} files".format(len(pos_data), fail_count))
  return pos_data, move_data, result_data, move_idxs


def get_training_curve_data(df, model_dir, pos_data, move_data, result_data, move_idxs, sgf_files, idx_start, eval_every):
  model_paths = get_model_paths(model_dir)
  player=None

  temp_values = [[move_data, result_data, move_idxs, sgf_files]]

  print("Evaluating models {}-{}, eval_every={}".format(idx_start, len(model_paths), eval_every))
  for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
    if "num" in df and idx in df["num"].values:
      print ("idx {} already processed => {}".format(idx, df[df['num'] == idx]))
      continue

    if player:
      restore_params(model_paths[idx], player)
    else:
      player = load_player(model_paths[idx])

    correct, squared_errors = eval_player(temp_values, player=player, positions=pos_data, moves=move_data, results=result_data)

    avg_acc = np.mean(correct)
    avg_mse = np.mean(squared_errors)
    print("Model: {}, acc: {:4f}, mse: {:4f}".format(model_paths[idx], avg_acc, avg_mse))
    df = df.append({"num": idx, "acc": avg_acc, "mse": avg_mse}, ignore_index=True)

  return df, temp_values


def exponential_moving_average(data, alpha = 0.2):
  averages = []
  avg = data[0] if len(data) > 0 else 0
  for value in data:
    avg = alpha * value + (1 - alpha) * avg
    averages.append(avg)
  return averages


def save_plots(data_dir, df):
  plt.plot(df["num"], df["acc"])
  plt.plot(df["num"], exponential_moving_average(df["acc"]))
  plt.xlabel("Model idx")
  plt.ylabel("Accuracy")
  plt.title("Accuracy in Predicting Professional Moves")
  plot_path = os.path.join(data_dir, "move_acc.png")
  plt.savefig(plot_path)

  plt.clf()

  plt.plot(df["num"], df["mse"])
  plt.plot(df["num"], exponential_moving_average(df["mse"]))
  plt.xlabel("Model idx")
  plt.ylabel("MSE/4")
  plt.title("MSE in predicting outcome")
  plot_path = os.path.join(data_dir, "value_mse.png")
  plt.savefig(plot_path)

  df = df.tail(40)
  df = df.reset_index(drop=True)

  plt.clf()

  plt.plot(df["num"], df["acc"])
  plt.plot(df["num"], exponential_moving_average(df["acc"]))
  plt.xlabel("Model idx")
  plt.ylabel("Accuracy")
  plt.title("Accuracy in Predicting Professional Moves")
  plot_path = os.path.join(data_dir, "move_acc2.png")
  plt.savefig(plot_path)

  plt.clf()

  plt.plot(df["num"], df["mse"])
  plt.plot(df["num"], exponential_moving_average(df["mse"]))
  plt.xlabel("Model idx")
  plt.ylabel("MSE/4")
  plt.title("MSE in predicting outcome")
  plot_path = os.path.join(data_dir, "value_mse2.png")
  plt.savefig(plot_path)


def load_checkpoint(save_file, params):
  loaded, data = oneoff_utils.load_checkpoint(save_file, params)
  if loaded:
    seed, df = data
    with pd.option_context("display.max_rows", 10):
      print(df)
  else:
    seed = np.random.randint(10000)
    df = pd.DataFrame()

  rand = np.random.RandomState(seed)
  print (rand.randint(100))

  return df, rand, seed


def main(unusedargv):
  data_dir = FLAGS.plot_dir
  checkpoint_save_file = os.path.join(data_dir, "training_curve_df")
  checkpoint_params = [FLAGS.idx_start, FLAGS.eval_every,
               FLAGS.komi, FLAGS.num_positions, FLAGS.sgf_dir]

  df, rand, seed = load_checkpoint(checkpoint_save_file, checkpoint_params)

  sgf_files = oneoff_utils.find_and_filter_sgf_files(FLAGS.sgf_dir, FLAGS.min_year, FLAGS.komi)
  pos_data, move_data, result_data, move_idxs = sample_positions_from_games(
      rand, sgf_files, FLAGS.num_positions)

  df, temp_values = get_training_curve_data(
    df, FLAGS.model_dir, pos_data, move_data, result_data, move_idxs, sgf_files,
    FLAGS.idx_start, eval_every=FLAGS.eval_every)
  df['num'] = df.num.astype(np.int64)

  print ("Saving plots", data_dir)
  save_plots(data_dir, df)

  pickle_path = os.path.join(data_dir, "test.pickle")
  with open(pickle_path, 'wb') as handle:
    pickle.dump(temp_values, handle)

  oneoff_utils.save_checkpoint(checkpoint_save_file, checkpoint_params, (seed, df))

FLAGS = tf.app.flags.FLAGS

if __name__ == "__main__":
  tf.app.run(main)
