"""Plot the percent of zero-ed weights of various tensors over a training run.

Example usage:
  mkdir -p data/zero_weights
  BOARD_SIZE=19 python3 oneoffs/zero_weights.py --base_dir "gs://minigo-pub/v7-19x19/"
"""
import sys
sys.path.insert(0, '.')

import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import re
from absl import app, flags
from collections import defaultdict
from tqdm import tqdm

import dual_net
import fsdb
import oneoff_utils

flags.DEFINE_string("plot_dir", "data/zero_weights", "Where to save the plots.")
flags.DEFINE_integer("idx_start", 200, "Only take models after given idx.")
flags.DEFINE_integer("eval_every", 10, "Eval every k models.")

FLAGS = flags.FLAGS


def reduce_var(var):
    return re.sub(r'_[0-9]+', '_<X>', var)


def reduce_and_print_vars(var_names):
    reduced_vars = sorted(set(map(reduce_var, var_names)))
    print('vars names({} reduced to {}):'.format(
        len(var_names), len(reduced_vars)))
    for v in reduced_vars:
        print('\t', v)
    return reduced_vars


def get_zero_weights_data(model_paths, idx_start, eval_every):

    print('Reading models {}-{}, eval_every={}'.format(
        idx_start, len(model_paths), eval_every))

    def zero_weights(tensor):
        return np.size(tensor), np.sum( tensor < 1e-10)

    var_names = tf.train.load_checkpoint(model_paths[1]) \
        .get_variable_to_dtype_map().keys()
    reduced_vars = reduce_and_print_vars(var_names)
    # Not a real var, sorry.
    reduced_vars.remove('global_step')

    df = pd.DataFrame()
    for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
        model_path = model_paths[idx]
        ckpt = tf.train.load_checkpoint(model_path)

        row = defaultdict(lambda: [0, 0])
        for v in var_names:
            count, zero = zero_weights(ckpt.get_tensor(v))
            row[v] = [zero, count]
            if reduce_var(v) != v:
                row[reduce_var(v)][0] += zero
                row[reduce_var(v)][1] += count

        row = dict(row)
        for k, v in row.items():
            row[k] = v[0] / v[1]

        row['model'] = idx
        df = df.append(row, ignore_index=True)
    return df


def save_plots(data_dir, df):
    for column in sorted(df.columns.values):
        if column == 'model':
            continue

        if '<X>' in column:
            plt.figure()
            plt.plot(df['model'].astype('int64'), df[column])
            plt.xlabel('Model idx')
            plt.ylabel('precent of weights that are 0')
            plt.title('{} zero weight ratio over run'.format(column))

            file_name = '{}.png'.format(column.replace('/', '-'))
            plot_path = os.path.join(data_dir, file_name)

            plt.savefig(plot_path)
            # Showing plots gets old really fast but can be uncommented if desired.
            # plt.show()
            plt.close()


def main(unusedargv):
    model_paths = oneoff_utils.get_model_paths(fsdb.models_dir())

    # Calculate zero weight ratios over a sequence of models.
    df = get_zero_weights_data(model_paths, FLAGS.idx_start, FLAGS.eval_every)
    print(df)
    save_plots(FLAGS.plot_dir, df)


if __name__ == "__main__":
    app.run(main)
