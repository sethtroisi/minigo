# Copyright 2019 Google LLC
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

"""
Used to select set of models for cross_eval_runs

Usage:
python cross_eval_models.py cross_eval_ratings_2019_01.csv

"""
import sys
sys.path.insert(0, '.')

import csv
import re
import json

# Number of models to select per run
MODELS_PER_RUN = 100
# Include this many of the best models (as determined by selfplay)
TOP_MODELS_PER_RUN = 25


def main():
    #assert len(sys.argv) == 3, sys.argv
    #path = sys.argv[1]
    in_path = 'cross_eval_ratings_2019_01_18.csv'
    out_path = 'cross_eval_models2.json'

    with open(in_path) as csv_file:
        data = csv.reader(csv_file)
        data = list(data)

    print (len(data), "lines of buckets and models")

    header = data.pop(0)
    assert header[0] == 'bucket', header

    name_range = {}
    run_models = {}
    while data[0][0] != 'model_id':
        run, low, high = data.pop(0)
        name_range[run] = range(int(low), int(high) + 1)
        run_models[run] = []

    header = data.pop(0)
    assert header[0] == 'model_id', header

    for row in list(data):
        model, name, rating = row
        model = int(model)
        rating = float(rating)
        if not re.match(r'^\d{6}', name):
            continue

        for run, model_range in name_range.items():
            if model in model_range:
                run_index = model - model_range.start
                run_models[run].append((run_index, name, rating))
                break
        else:
            print ('No range', row)

    print()
    for run, models in sorted(run_models.items()):
        if len(models):
            models.sort()
            print(run, len(models), models[0], 'to', models[-1])
    print()

    all_names = {}
    keep = []
    # only interested in some runs
    for version in range(9, 16):
        if version == 11:
            # v11 failed
            continue

        run = 'v{}-19x19'.format(version)
        assert run in run_models, run
        models = run_models[run]

        included = []

        best_models = sorted([(r, i, n) for i, n, r in models], reverse=True)
        best_models = best_models[:TOP_MODELS_PER_RUN]
        for r, i, n in best_models:
            included.append(n)
            models.remove((i, n, r))

        count = MODELS_PER_RUN - TOP_MODELS_PER_RUN
        for k in range(count):
            interval_start = k * len(models) // count
            interval_end = (k+1) * len(models) // count

            # Choose the best rated model from a window of models so that the
            # curve for the model is high waterish marks
            to_consider = models[interval_start : interval_end]
            best = max((r, i, n) for (i, n, r) in to_consider)
            best_name = best[2]
            assert best_name not in all_names, (
                best_name, version, all_names[best_name])

            included.append(best_name)
            all_names[best_name] = version


        included.sort()
        for model in included:
            keep.append((run, model))

        print(run, len(included))
        print('\t', included[:5])
        print('\t ...', included[-5:])
        assert len(included) == MODELS_PER_RUN

    with open(out_path, 'w') as out_file:
        json.dump(keep, out_file)

if __name__ == '__main__':
    main()
