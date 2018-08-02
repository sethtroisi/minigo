"""Analyse if any variable correlate with model eval.

Example usage:
  python3 oneoffs/tensorboard_verus_eval.py --base_dir "v9-19x19/work_dir"
"""
import sys
sys.path.insert(0, '.')

from collections import defaultdict, Counter
from glob import glob
import os.path
import urllib.request
import json

from absl import app, flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm

flags.DEFINE_string('base_dir', None, 'Directory of tf events')

FLAGS = flags.FLAGS


TEMP = """model.ckpt-12|001
model.ckpt-27|002
model.ckpt-69|003
model.ckpt-138|004
model.ckpt-234|005
model.ckpt-358|006
model.ckpt-510|007
model.ckpt-691|008
model.ckpt-900|009
model.ckpt-1138|010
model.ckpt-1405|011
model.ckpt-1702|012
model.ckpt-2036|013
model.ckpt-2462|014
model.ckpt-2921|015
model.ckpt-3412|016
model.ckpt-3933|017
model.ckpt-4487|018
model.ckpt-5073|019
model.ckpt-5692|020
model.ckpt-6341|021
model.ckpt-7021|022
model.ckpt-7733|023
model.ckpt-8478|024
model.ckpt-9252|025
model.ckpt-10063|026
model.ckpt-10902|027
model.ckpt-11779|028
model.ckpt-12687|029
model.ckpt-13631|030
model.ckpt-14611|031
model.ckpt-15625|032
model.ckpt-16649|033
model.ckpt-17673|034
model.ckpt-18697|035
model.ckpt-19721|036
model.ckpt-20745|037
model.ckpt-21769|038
model.ckpt-22793|039
model.ckpt-23817|040
model.ckpt-24841|041
model.ckpt-25865|042
model.ckpt-26889|043
model.ckpt-27913|044
model.ckpt-28937|045
model.ckpt-29961|046
model.ckpt-30985|047
model.ckpt-32009|048
model.ckpt-33033|049
model.ckpt-34057|050
model.ckpt-35081|051
model.ckpt-36105|052
model.ckpt-37129|053
model.ckpt-38153|054
model.ckpt-39177|055
model.ckpt-40201|056
model.ckpt-41225|057
model.ckpt-42249|058
model.ckpt-43273|059
model.ckpt-44297|060
model.ckpt-45321|061
model.ckpt-46345|062
model.ckpt-47369|063
model.ckpt-48393|064
model.ckpt-49417|065
model.ckpt-50441|066
model.ckpt-51465|067
model.ckpt-52489|068
model.ckpt-53513|069
model.ckpt-54537|070
model.ckpt-55561|071
model.ckpt-56585|072
model.ckpt-57609|073
model.ckpt-58633|074
model.ckpt-59657|075
model.ckpt-60681|076
model.ckpt-61705|077
model.ckpt-62729|078
model.ckpt-63753|079
model.ckpt-64777|080
model.ckpt-65801|081
model.ckpt-66825|082
model.ckpt-67849|083
model.ckpt-68873|084
model.ckpt-69897|085
model.ckpt-70921|086
model.ckpt-71945|087
model.ckpt-72969|088
model.ckpt-73993|089
model.ckpt-75017|090
model.ckpt-76041|091
model.ckpt-77065|092
model.ckpt-78089|093
model.ckpt-79113|094
model.ckpt-80137|095
model.ckpt-81161|096
model.ckpt-82185|097
model.ckpt-83209|098
model.ckpt-84233|099
model.ckpt-85257|100"""

def correlate(events, evals):
    global TEMP
    TEMP = { int(ckpt.split('-')[-1]): int(m)
        for ckpt, m in [l.split('|') for l in TEMP.split("\n")]}

    print("{} events, {} evals".format(len(events), len(evals)))
    # events * x = evals
    eval_vector = []
    events_vector = []
    value_count = Counter()
    parts = ['value_cost', 'policy_cost', 'policy_accuracy_top_1',
             'policy_accuracy_top_3', 'value_confidence']
    for ckpt, values in sorted(events.items()):
        for v in values.keys():
            value_count[v] += 1
        if ckpt not in TEMP:
            continue

        if all(p in values for p in parts):
            print (ckpt, TEMP[ckpt])
            eval_vector.append([elo for m,elo,var in evals if m == TEMP[ckpt]][0])
            events_vector.append([values[p] for p in parts])


    print (events_vector)
    print (eval_vector)

    print(value_count)
    print()
    print(len(events_vector), len(eval_vector))

    res = np.linalg.lstsq(events_vector, eval_vector, rcond=None)
    residuals = res[1]
    print (res[0])
    print (residuals)

    r2 = 1 - residuals / (len(eval_vector) * np.var(eval_vector))
    print(r2)

def load_evals():
    # query eval from cloudygo.com
    url = "http://cloudygo.com/v9-19x19/json/ratings.json"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_all_events(base_dir):
    """Get stats by checkpoint"""
    # We see lots of duplicate entries, just overridding for now
    checkpoint_stats = defaultdict(dict)

    for event_file in sorted(glob(os.path.join(base_dir, '*tfevents*'))):
        records = list(tf.train.summary_iterator(event_file))
        print(os.path.basename(event_file), len(records))
        for record in records:
            what = record.WhichOneof("what")
            #if what not in ("meta_graph_def", "graph_def"):
            if what == "summary":
                assert len(str(record)) <= 400, len(str(record))
                assert len(record.summary.value) == 1, record
                value = record.summary.value[0]
                which_value = value.WhichOneof("value")
                assert which_value == "simple_value", which_value

                #if value.tag in checkpoint_stats[record.step]:
                checkpoint_stats[record.step][value.tag] = value.simple_value

        print(len(checkpoint_stats))
        print()
    return checkpoint_stats

def main(unusedargv):
    evals = load_evals()
    print (evals)
    events = get_all_events(FLAGS.base_dir)
    correlate(events, evals)

if __name__ == "__main__":
    app.run(main)
