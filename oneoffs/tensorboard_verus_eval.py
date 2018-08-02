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


TEMP = {138: 4, 234: 5, 358: 6, 510: 7, 691: 8, 900: 9, 1138: 10, 1405: 11, 1702: 12, 2036: 13, 2462: 14, 2921: 15, 3412: 16, 3933: 17, 4487: 18, 5073: 19, 5692: 20, 6341: 21, 7021: 22, 7733: 23, 8478: 24, 9252: 25, 10063: 26, 10902: 27, 11779: 28, 12687: 29, 13631: 30, 14611: 31, 15625: 32, 16649: 33, 17673: 34, 18697: 35, 19721: 36, 20745: 37, 21769: 38, 22793: 39, 23817: 40, 24841: 41, 25865: 42, 26889: 43, 27913: 44, 28937: 45, 29961: 46, 30985: 47, 32009: 48, 33033: 49, 34057: 50,
35081: 51, 36105: 52, 37129: 53, 38153: 54, 39177: 55, 40201: 56, 41225: 57, 42249: 58, 43273: 59, 44297: 60, 45321: 61, 46345: 62, 47369: 63, 48393: 64, 49417: 65, 50441: 66, 51465: 67, 52489: 68, 53513: 69, 54537: 70, 55561: 71, 56585: 72, 57609: 73, 58633: 74, 59657: 75, 60681: 76, 61705: 77, 62729: 78, 63753: 79, 64777: 80, 65801: 81, 66825: 82, 67849: 83, 68873: 84, 69897: 85, 70921: 86, 71945: 87, 72969: 88, 73993: 89, 75017: 90, 76041: 91, 77065: 92, 78089: 93, 79113: 94, 80137:
95, 81161: 96, 82185: 97, 83209: 98, 84233: 99, 85257: 100, 86281: 101, 87305: 102, 88329: 103, 89353: 104, 90377: 105, 91401: 106, 92425: 107, 93449: 108, 94473: 109, 95497: 110, 96521: 111, 97545: 112, 98569: 113, 99593: 114, 100617: 115, 101641: 116, 102665: 117, 103689: 118, 104713: 119, 105737: 120, 106761: 121, 107785: 122, 108809: 123, 109833: 124, 110857: 125, 111881: 126, 112905: 127, 113929: 128, 114953: 129, 115977: 130, 117001: 131, 118025: 132, 119049: 133, 120073:
134, 121097: 135, 122121: 136, 123145: 137, 124169: 138, 125193: 139, 126217: 140, 127241: 141, 128265: 142, 129289: 143, 130313: 144, 131337: 145, 132361: 146, 133385: 147, 134409: 148, 135433: 149, 136457: 150, 137481: 151, 138505: 152, 139529: 153, 140553: 154, 141577: 155, 142601: 156, 143625: 157, 144649: 158, 145673: 159, 146697: 160, 147721: 161, 148745: 162, 149769: 163, 150793: 164, 151817: 165, 152841: 166, 153865: 167, 154889: 168, 155913: 169, 156937: 170, 157961: 171,
158985: 172, 160009: 173, 161033: 174, 162057: 175, 163081: 176, 164105: 177, 165129: 178, 166153: 179, 167177: 180, 168201: 181, 169225: 182, 170249: 183, 171273: 184, 172297: 185, 173321: 186, 174345: 187, 175369: 188, 176393: 189, 177417: 190, 178441: 191, 179465: 192, 180489: 193, 181513: 194, 182537: 195, 183561: 196, 184585: 197, 185609: 198, 186633: 199, 187657: 200}


def correlate(events, evals):
    print("{} events, {} evals".format(len(events), len(evals)))
    # events * x = evals
    eval_vector = []
    events_vector = []
    value_count = Counter()
    parts = ['value_cost', 'policy_cost', 'policy_accuracy_top_1',
             'policy_accuracy_top_3', 'value_confidence', 'policy_entropy']
    for ckpt, values in sorted(events.items()):
        for v in values.keys():
            value_count[v] += 1
        if ckpt not in TEMP:
            continue

        if all(p in values for p in parts):
            print (ckpt, TEMP[ckpt])
            eval_vector.append([elo for m,elo,var in evals if m == TEMP[ckpt]][0])
            events_vector.append([values[p] for p in parts])


#    print (events_vector)
#    print (eval_vector)

    print(value_count)
    print()
    print(len(events_vector), len(eval_vector))

    res = np.linalg.lstsq(events_vector, eval_vector, rcond=None)
    residuals = res[1]
    print (parts)
    print (res[0])

    #print (residuals)
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

    max_temp = max(TEMP.keys())

    for event_file in sorted(glob(os.path.join(base_dir, '*tfevents*'))):
        records = list(tf.train.summary_iterator(event_file))
        print(os.path.basename(event_file), len(records))
        for record in records:
            if record.step > max_temp:
                break
            if record.step not in TEMP:
                continue

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
