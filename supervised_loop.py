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

"""Runs a Supervised learning loop locally."""

import os
import sys

import preprocessing
import dual_net
import main


def supervised():
    """Run the reinforcement learning loop

    This is meant to be more of an integration test than a realistic way to run
    the reinforcement learning.
    """

    assert len(sys.argv) >= 4, ("Three Args: layers, filters, machine | got:", sys.argv[1:])
    layers, filters, machine = map(int, sys.argv[1:4])

    value_mult = 1.0
    if len(sys.argv) >= 5:
        value_mult = int(sys.argv[4])


    # monkeypatch the hyperparams so that we get a quickly executing network.
    hyperparams = dual_net.get_default_hyperparams()
    hyperparams.update({
        'num_shared_layers': layers,
        'k': filters,
        'fc_width': 2 * filters,
        'value_head_loss_scalar': 1.0 / value_mult,
    })
    dual_net.get_default_hyperparams = lambda: hyperparams

    #dual_net.TRAIN_BATCH_SIZE = 16
    #dual_net.EXAMPLES_PER_GENERATION = 64

    #monkeypatch the shuffle buffer size so we don't spin forever shuffling up positions.
    preprocessing.SHUFFLE_BUFFER_SIZE = 200000

    pro_dir = "data/records"
    holdout_dir = "data/holdouts"

    name = "{}-{}-{}".format(layers, filters, machine)
    if value_mult != 1.0:
        name = "{}-{}-{}-{}".format(layers, filters, value_mult, machine)

    base_dir = "supervised/{}/".format(name)
    working_dir = os.path.join(base_dir, 'models_in_training')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(base_dir)
    os.makedirs(working_dir)
    os.makedirs(model_dir)

    print("Creating random initial weights...")
    bootstrap_save_path = os.path.join(model_dir, '000000-bootstrap')
    main.bootstrap(working_dir, bootstrap_save_path)

    for generation in [1,10,50,100,200,400]:
        model_name = "{:06d}-supervised-{}x{}-{}".format(
            generation, layers, filters, machine)
        print("Training {}!".format(model_name))
        model_save_path = os.path.join(model_dir, model_name)
        main.train_dir(working_dir, pro_dir, model_save_path, 30000 * generation)

        print("Validate on 'holdout' data")
        main.validate(working_dir, holdout_dir, checkpoint_name=model_save_path, validate_name="test")


if __name__ == '__main__':
    supervised()
