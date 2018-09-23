#!/usr/bin/env python

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

import os
import pickle

from absl import app, flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import dual_net
import features as features_lib
import sgf_wrapper

flags.DEFINE_string("sgf_root", None, "root directory for eval games")

flags.DEFINE_integer("first", 20, "first move in game to consider")
flags.DEFINE_integer("last", 150, "last move in game to consider")
flags.DEFINE_integer("every", 10, "choose every X position from game")

flags.DEFINE_integer("embedding_size", 361, "Size of embedding")

flags.mark_flag_as_required('sgf_root')

FLAGS = flags.FLAGS

EMBEDDING_SIZE = 361

def chunks(l, n):
  for i in range(0, len(l), n):
      yield l[i:i + n]


def get_files():
  root = '/media/eights/Sojourner/minigo-data/v10-19x19/eval/'
  files = []
  for d in os.listdir(root):
    for f in os.listdir(os.path.join(root, d))[:5000]:
        if f.endswith('.sgf'):
            files.append(os.path.join(root, d, f))
  return files


dual_net.flags.FLAGS([""])
features, labels = dual_net.get_inference_input()
p_out, v_out, logits, shared = dual_net.model_inference_fn(features, False)
predictions = {
    'shared': shared
}

sess = tf.Session()
save_file = "saved_models/000721-eagle"
tf.train.Saver().restore(sess, save_file)

try:
  progress = tqdm(get_files())
  embeddings = np.empty([len(progress), EMBEDDING_SIZE])
  metadata = []
  for i, f in enumerate(progress):
    short_f = os.path.basename(f)
    progress.set_description("Processing %s" % short_f)

    processed = []
    for idx, p in enumerate(sgf_wrapper.replay_sgf_file(f)):
      if idx < FLAGS.first: continue
      if idx > FLAGS.last: break
      if idx % FLAGS.every != 0: continue

      processed.append(features_lib.extract_features(p.position))
      metadata.append((f, idx))

    if len(processed) > 0:
      # If len(processed) gets large have to chunk.
      res = sess.run(predictions, feed_dict={features: processed})
      for r in res['shared']:
        assert np.size(r) == EMBEDDING_SIZE, np.size(r)
        embeddings[i] = r.flatten()

except:
  raise
finally:
  with open('embeddings.pickle', 'wb') as pickle_file:
    pickle.dump([metadata, embeddings], pickle_file)
