

import json
import os
import re
import shutil

import sqlite3

with sqlite3.connect("ratings.db") as db:
    sql_models = db.execute("select model_name from models").fetchall()
sql_models = [m[0] for m in sql_models]

with open("oneoffs/cross_eval_models.json") as f:
    run_models = json.load(f)

print (len(sql_models), "sql models")
run_names = [m[1] for m in run_models]
print (len(run_models), "run models", len(run_names))
#print ([m for m in run_names if run_names.count(m) > 1])

#print ("example sql:", sql_models[0])
print ("example model:", run_models[0])

run_names = set(m[1] for m in run_models)
extra_sql = set(sql_models) - run_names
assert len(extra_sql) == 0, extra_sql

# do something with all cross-eval files and move them

short_names = set()
for m in run_models:
    short = m[0].split('-')[0] + "-" + str(int(m[1].split('-')[0]))
    short_names.add(short)

assert len(short_names) == len(run_models)

cross_eval_dir = "../cloudygo/instance/data/cross-eval/eval/"
models_re = re.compile('minigo-cc-evaluator-(.*)-vs-(.*)-[bw]{2}-')

count_match = {True: 0, False: 0, "run": 0}
for root, _, files in os.walk(cross_eval_dir):
#    if not re.search(r'2019-01-1[0-9]', root):
#        continue

#        assert parts, root
#

#            if not (parts.group(1) in short_names and parts.group(2) in short_names):
#                count_match["run"] += 1
#                f = os.path.basename(root) + "/" + name
#                print("gsutil mv gs://tensor-go-minigo-cross-evals/sgf/eval/" + f + " "
#                                "gs://tensor-go-minigo-cross-evals/sgf/eval_temp/" + f)

    for name in files:
        parts = models_re.search(name)
        count_match[parts is not None] += 1

        '''
        if parts:
            if parts.group(1) in short_names and parts.group(2) in short_names:
                count_match["run"] += 1

                new_dir = root.replace('cross-eval', 'cross-run-eval')
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                    print ("new_dir:", new_dir)

                dst = os.path.join(new_dir, name)
                if not os.path.isfile(dst):
                    src = os.path.join(root, name)
                    print ("copy", name, "\t", root)
        '''
print ("matches:", count_match)
