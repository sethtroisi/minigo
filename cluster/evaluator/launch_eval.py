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

import sys
sys.path.insert(0, '.')

import json
import os
import re
import time
from collections import Counter

import fire
from absl import flags
import kubernetes
import yaml

from rl_loop import fsdb
from ratings import ratings


def launch_eval_job(m1_path, m2_path, job_name, bucket_name, completions=5):
    """Launches an evaluator job.
    m1_path, m2_path: full gs:// paths to the .pb files to match up
    job_name: string, appended to the container, used to differentiate the job
    names (e.g. 'minigo-cc-evaluator-v5-123-v7-456')
    bucket_name: Where to write the sgfs, passed into the job as $BUCKET_NAME
    completions: the number of completions desired
    """
    if not all([m1_path, m2_path, job_name, bucket_name]):
        print("Provide all of m1_path, m2_path, job_name, and bucket_name "
              "params")
        return

    def isGsPath(path):
        return path.startswith('gs://')

    assert isGsPath(m1_path), 'm1_path: ' + m1_path + ' must start gs://'
    assert isGsPath(m2_path), 'm2_path: ' + m2_path + ' must start gs://'
    assert not isGsPath(bucket_name), 'bucket_name must not be gs:// path'

    api_instance = get_api()

    raw_job_conf = open("cluster/evaluator/cc-evaluator.yaml").read()

    os.environ['BUCKET_NAME'] = bucket_name

    os.environ['MODEL_BLACK'] = m1_path
    os.environ['MODEL_WHITE'] = m2_path
    os.environ['JOBNAME'] = job_name + '-bw'
    env_job_conf = os.path.expandvars(raw_job_conf)

    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions

    resp_bw = api_instance.create_namespaced_job('default', body=job_conf)

    os.environ['MODEL_WHITE'] = m1_path
    os.environ['MODEL_BLACK'] = m2_path
    os.environ['JOBNAME'] = job_name + '-wb'
    env_job_conf = os.path.expandvars(raw_job_conf)
    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions

    resp_wb = api_instance.create_namespaced_job('default', body=job_conf)
    return job_conf, resp_bw, resp_wb


def same_run_eval(black_num=0, white_num=0):
    """Shorthand to spawn a job matching up two models from the same run,
    identified by their model number """
    if black_num <= 0 or white_num <= 0:
        print("Need real model numbers")
        return

    b = fsdb.get_model(black_num)
    w = fsdb.get_model(white_num)

    b_model_path = os.path.join(fsdb.models_dir(), b)
    w_model_path = os.path.join(fsdb.models_dir(), w)

    return launch_eval_job(b_model_path + ".pb",
                           w_model_path + ".pb",
                           "{}-{}".format(black_num, white_num),
                           flags.FLAGS.bucket_name)


def cross_run_eval(run_a, model_a, run_b, model_b):
    """Shorthand to spawn a job matching up two models from the different run,
    identified by their bucket and model number """

    run_a = run_a.replace('-19x19', '')
    run_b = run_b.replace('-19x19', '')
    assert len(run_a) in (2,3) and len(run_b) in (2,3), (run_a, run_b)
    model_re = re.compile(r'^[0-9]{6}-[a-z-]*$')
    assert model_re.match(model_a), model_a
    assert model_re.match(model_b), model_b

    num_a = int(model_a.split('-')[0])
    num_b = int(model_b.split('-')[0])
    assert 0 <= num_a <= 1000, num_a
    assert 0 <= num_b <= 1000, num_b

    PROJECT = os.environ.get("PROJECT")
    bucket_a = "gs://{}-minigo-{}-19".format(PROJECT, run_a)
    bucket_b = "gs://{}-minigo-{}-19".format(PROJECT, run_b)

    path_a = os.path.join(bucket_a, 'models', model_a + ".pb")
    path_b = os.path.join(bucket_b, 'models', model_b + ".pb")

    tag = '{}-{}-vs-{}-{}'.format(run_a, num_a, run_b, num_b)

    cross_eval_bucket = PROJECT + '-minigo-cross-evals'
    return launch_eval_job(path_a, path_b, tag, cross_eval_bucket, 2)


def _append_pairs(new_pairs, dry_run):
    desired_pairs = restore_pairs() or []
    desired_pairs += new_pairs
    print("Adding {} new pairs, queue has {} pairs".format(len(new_pairs), len(desired_pairs)))
    if not dry_run:
        save_pairs(desired_pairs)


def add_uncertain_pairs(dry_run=False):
    new_pairs = ratings.suggest_pairs()
    _append_pairs(new_pairs, dry_run)



def get_cross_eval_pairs():
    #def cross_run_eval(run_a, model_a, run_b, model_b):

    all_models = read_cross_run_models()
    print(len(all_models), "cross eval models")

    # key in ratings.db
    model_ids = {m[1]: ratings.model_id(m[1]) for m in all_models}
    assert len(model_ids) == len(all_models), "Duplicate model name!"
#    assert len(model_ids) == len(set(model_ids.values())), "Duplicate row!"

    raw_scores = ratings.compute_ratings()
    r = {ratings.model_name_for(k): v[0] for k, v in raw_scores.items()}
    print (len(r), "ratings")

    game_counts = Counter(ratings.wins_subset(fsdb.models_dir()))
    print ("Found", sum(game_counts.values()), "games")

    # priority is roughly related to expected gain of information
    pairs = []
    for run_a, model_a in all_models:
        for run_b, model_b in all_models:
            if (run_a, model_a) >= (run_b, model_b):
                continue

            # if not present use model_num as rating which helps sparse rating.
            r_a = r.get(model_a, int(model_a.split('-')[0]))
            r_b = r.get(model_b, int(model_b.split('-')[0]))

            win_prob = 1 / (1 + 10 ** (-(r_a - r_b)/400))
            variance = win_prob * (1 - win_prob)

            # count of head to head encounters
            games = (game_counts[(model_ids[model_a], model_ids[model_b])] +
                     game_counts[(model_ids[model_b], model_ids[model_a])])

            # Controls how much to explore unique pairs verus equal strength.
            power = 2.5
            priority = variance  / (1 + games) ** power

            pairs.append((priority, (run_a, model_a, run_b, model_b)))

    pairs.sort(reverse=True)
    pairs = pairs[:10]
    for priority, pair in pairs:
        print ("Consider priority:", priority, "pair:", pair)

    existing_pairs = restore_pairs() or []
    pairs = [pair for _, pair in pairs if pair not in existing_pairs]

    _append_pairs(pairs, dry_run=False)



def add_top_pairs(dry_run=False):
    """ Pairs up the top twenty models against each other.
    #1 plays 2,3,4,5, #2 plays 3,4,5,6 etc. for a total of 15*4 matches.
    """
    top = ratings.top_n(10)
    new_pairs = []
    for idx, t in enumerate(top[:5]):
        new_pairs += [[t[0], o[0]] for o in top[idx+1:idx+5]]
    print(new_pairs)
    _append_pairs(new_pairs, dry_run)


def zoo_loop(sgf_dir=None, max_jobs=40):
    """Manages creating and cleaning up match jobs.

    - Load whatever pairs didn't get queued last time, and whatever our most
      recently seen model was.
    - Loop and...
        - If a new model is detected, create and append new pairs to the list
        - Automatically queue models from a list of pairs to keep a cluster
          busy
        - As jobs finish, delete them from the cluster.
        - If we crash, write out the list of pairs we didn't manage to queue

    sgf_dir -- the directory where sgf eval games should be used for computing
      ratings.
    max_jobs -- the maximum number of concurrent jobs.  jobs * completions * 2
      should be around 500 to keep kubernetes from losing track of completions
    """
    desired_pairs = restore_pairs() or []
    last_model_queued = restore_last_model()

    if sgf_dir:
        sgf_dir = os.path.abspath(sgf_dir)

    api_instance = get_api()
    try:
        while True:
            last_model = fsdb.get_latest_pb()[0]
            if last_model_queued < last_model:
                print("Adding models {} to {} to be scheduled".format(
                    last_model_queued+1, last_model))
                for m in reversed(range(last_model_queued+1, last_model+1)):
                    desired_pairs += make_pairs_for_model(m)
                last_model_queued = last_model
                save_last_model(last_model)

            cleanup(api_instance)
            r = api_instance.list_job_for_all_namespaces()
            if len(r.items) < max_jobs:
                if len(desired_pairs) == 0:
                    if sgf_dir:
                        print("Out of pairs!  Syncing new eval games...")
                        ratings.sync(sgf_dir)
                        print("Updating ratings and getting suggestions...")
                        add_uncertain_pairs()
                        desired_pairs = restore_pairs() or []
                        print("Got {} new pairs".format(len(desired_pairs)))
                        print(ratings.top_n())
                    else:
                        print("Out of pairs!  Sleeping")
                        time.sleep(300)
                        continue

                next_pair = desired_pairs.pop()  # take our pair off
                print("Enqueuing:", next_pair)
                try:
                    same_run_eval(*next_pair)
                except:
                    desired_pairs.append(next_pair)
                    raise
                save_pairs(sorted(desired_pairs))
                save_last_model(last_model)
                time.sleep(6)

            else:
                print("{}\t{} jobs outstanding. ({} to be scheduled)".format(
                      time.strftime("%I:%M:%S %p"),
                      len(r.items), len(desired_pairs)))
                time.sleep(60)
    except:
        print("Unfinished pairs:")
        print(sorted(desired_pairs))
        save_pairs(sorted(desired_pairs))
        save_last_model(last_model)
        raise


def cross_run_eval_matchmaker_loop(sgf_dir, max_jobs=20):
    """Manages creating and cleaning up cross bucket evaluation jobs.

    sgf_dir -- the directory where sgf eval games should be used for computing
      ratings.
    max_jobs -- the maximum number of concurrent jobs.  jobs * completions * 2
      should be around 200 to keep kubernetes from losing track of completions
    """
    desired_pairs = restore_pairs() or []

    sgf_dir = os.path.abspath(sgf_dir)

    api_instance = get_api()
    try:
        while True:
            cleanup(api_instance)
            r = api_instance.list_job_for_all_namespaces()
            if len(r.items) >= max_jobs:
                print("{}\t{} jobs outstanding. ({} to be scheduled)".format(
                      time.strftime("%I:%M:%S %p"),
                      len(r.items), len(desired_pairs)))
                time.sleep(60)
            else:
                if len(desired_pairs) == 0:
                    if sgf_dir:
                        print("Out of pairs!  Syncing new eval games...")
                        ratings.sync(sgf_dir)
                        print("Updating ratings and getting suggestions...")
                        get_cross_eval_pairs()
                        desired_pairs = restore_pairs() or []
                        print("Got {} new pairs".format(len(desired_pairs)))
                    else:
                        print("Out of pairs!  Sleeping")
                        time.sleep(300)
                        continue

                next_pair = desired_pairs.pop()  # take our pair off
                print("Queue", len(desired_pairs), "items")
                print("Enqueuing:", next_pair)
                try:
                    cross_run_eval(*next_pair)
                except:
                    desired_pairs.append(next_pair)
                    raise
                save_pairs(sorted(desired_pairs))
                time.sleep(6)
    except:
        print("Unfinished pairs:")
        print(sorted(desired_pairs))
        save_pairs(sorted(desired_pairs))
        raise

def read_json(filename):
    with open(filename) as f:
        return json.loads(f.read())

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def restore_pairs():
    return read_json('pairlist.json')

def save_pairs(pairs):
    write_json('pairlist.json', pairs)

def restore_last_model():
    return read_json('last_model.json')

def save_last_model(model):
    write_json('last_model.json', model)

def read_cross_run_models():
    return read_json('oneoffs/cross_eval_models.json')

def get_api():
    kubernetes.config.load_kube_config(persist_config=True)
    configuration = kubernetes.client.Configuration()
    return kubernetes.client.BatchV1Api(
        kubernetes.client.ApiClient(configuration))


def cleanup(api_instance=None):
    """ Remove completed jobs from the cluster """
    api = api_instance or get_api()
    r = api.list_job_for_all_namespaces()
    delete_opts = kubernetes.client.V1DeleteOptions()
    for job in r.items:
        if job.status.succeeded == job.spec.completions:
            print(job.metadata.name, "finished!")
            api.delete_namespaced_job(
                job.metadata.name, 'default', body=delete_opts)


def make_pairs_for_model(model_num=0):
    """ Create a list of pairs of model nums; play every model nearby, then
    every other model after that, then every fifth, etc.

    Returns a list like [[N, N-1], [N, N-2], ... , [N, N-12], ... , [N, N-50]]
    """
    if model_num == 0:
        return
    pairs = []
    pairs += [[model_num, model_num - i]
              for i in range(1, 5) if model_num - i > 0]
    pairs += [[model_num, model_num - i]
              for i in range(5, 71, 10) if model_num - i > 0]
    return pairs


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    fire.Fire({
        'zoo_loop': zoo_loop,
        'cross_run_eval_matchmaker_loop': cross_run_eval_matchmaker_loop,
        'same_run_eval': same_run_eval,
        'cross_run_eval': cross_run_eval,
        'cleanup': cleanup,
        'add_top_pairs': add_top_pairs,
        'launch_eval_job': launch_eval_job,
    }, remaining_argv[1:])
