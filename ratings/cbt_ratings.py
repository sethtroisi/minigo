### TODO header
"""


Inspiration from Seth's experience with CloudyGo db.

Usage:

$ export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/usr/share/grpc/roots.pem
$ sqlite3 cbt_ratings.db < ratings/schema.sql
$ python3 ratings/cbt_ratings.py  \
    --cbt_project "$PROJECT" \
    --cbt_instance "$CBT_INSTANCE" \
"""

import sys
sys.path.insert(0, '.')

import re
import math
from collections import defaultdict, Counter

import choix
import numpy as np
import sqlite3
from absl import flags
from tqdm import tqdm
from google.cloud import bigtable

from bigtable_input import METADATA, TABLE_STATE


FLAGS = flags.FLAGS

flags.mark_flags_as_required([
    "cbt_project", "cbt_instance"
])

MODEL_REGEX = re.compile("(\d*)-(.*)")
CROSS_EVAL_REGEX = re.compile("(v\d+)-(\d+)-vs-(v\d+)-(\d+)")
MODELS_FROM_FN = re.compile("(\d{6}-[a-z-]+)-(\d{6}-[a-z-]+)$")

def assert_pair_matches(pb, pw, m1, m2, sgf_file):
    if pb == m1 and pw == m2:
        return False
    if pb == m2 and pw == m1:
        return True
    assert False, ((pb, pw), (m1, m2), sgf_file)



def determine_model_id(sgf_file, pb, pw, model_runs):
    """Determine which run+model PB and PW are."""

    # Remove leading zeros
    num_b = str(int(pb.split('-')[0]))
    num_w = str(int(pw.split('-')[0]))

    # Possible runs for white and black player.
    runs_pb = model_runs[pb]
    runs_pw = model_runs[pw]
    assert runs_pb and runs_pw, (row_key, pb, pw)

    # Validation that cbt pb/pw match the filename.
    models = MODELS_FROM_FN.search(sgf_file)
    assert models, sgf_file
    m_1, m_2 = models.groups()
    assert_pair_matches(pb, pw, m_1, m_2, sgf_file)

    simple = CROSS_EVAL_REGEX.search(sgf_file)
    if not simple:
        return None
    run_1, num_1, run_2, num_2 = simple.groups()

    # We have to unravel a mystery here.
    # The file name tells up which number <=> run
    # The file name also tells us model_name (model_number + model_name)
    # pb,pw are model_name from PB[] and PW[].
    #
    # The easy case is num_1 != num_2, PB tells us model_number tells us run
    # The hard case is num_1 == num_2, this requires us checking if
    # the PB/PW match only one of the runs.

    to_consider = {run_1, run_2}
    runs_pb = set(runs_pb) & to_consider
    runs_pw = set(runs_pw) & to_consider
    assert runs_pb and runs_pw

    # To simply code assume run_1 goes with pb.
    # If not set swap = True

    if num_1 != num_2:
        swap = assert_pair_matches(num_b, num_w, num_1, num_2, sgf_file)
    else:
        # Imagine v12-80-vs-v10-80-bw-nhh5x-0-000080-duke-000080-duke
        # No way to tell which 80-duke is PB or PW
        assert pw != pb, (sgf_file)

        if len(runs_pb) == 1:
            swap = run_2 in runs_pb
        elif len(runs_pw) == 1:
            swap = run_1 in runs_pw
        else:
            # This would be very unlucky, both runs would have to have the same
            # model names for both numbers.
            assert False, (sgf_file, runs_pb, runs_pw)

    if swap:
        run_1, num_1, run_2, num_2 = \
            run_2, num_2, run_1, num_1

    assert num_b == num_1 and run_1 in runs_pb
    assert num_w == num_2 and run_2 in runs_pw

    # Verify the inverse isn't also valid.
    assert num_b == num_1 and run_1 in runs_pb
    assert num_w == num_2 and run_2 in runs_pw

    # (run_b, run_b)
    return run_1, run_2


def setup_models(models_table):
    """
    Read all (~10k) models from cbt and db
    Merge and write and new models to db.

    Returns:
      {(<run>,<model_name>): db_model_id}, {model_name: [run_a, run_b]}
    """

    model_ids = {}
    model_runs = defaultdict(list)

    with sqlite3.connect("cbt_ratings.db") as db:
        cur = db.execute("SELECT id, model_name, bucket FROM models")
        for model_id, model_name, run in cur.fetchall():
            assert model_id not in model_ids
            model_ids[(run, model_name)] = model_id
        print("Existing(db):", len(model_ids))

        cbt_models = 0
        new_models = []
        for row in tqdm(models_table.read_rows()):
            cbt_models += 1
            name = row.cell_value(METADATA, b'model').decode()
            run  = row.cell_value(METADATA, b'run').decode()
            num  = int(row.cell_value(METADATA, b'model_num').decode())
            assert run not in model_runs[name], (name, run)
            model_runs[name].append(run)

            if (run, name) not in model_ids:
                new_models.append((name, run, num))

        print("Existing(cbt):", cbt_models)

        if new_models:
            print("New for db:", len(new_models))

            db.executemany(
                  """INSERT INTO models VALUES (
                  null, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0)""",
                  new_models)

            # Read from db to pick up updates
            cur = db.execute("SELECT id, model_name, bucket FROM models")
            for model_id, model_name, run in cur.fetchall():
                model_ids[(run, model_name)] = model_id

        assert len(model_ids) == cbt_models
        return model_ids, model_runs


def sync(bt_table, model_ids, model_runs):
    # TODO(sethtroisi): Potentially only update from a starting rows.

    bucket = "???"
    print("Importing for bucket:", bucket)
    new_games = 0
    status = Counter()

    with sqlite3.connect("cbt_ratings.db") as db:
        c = db.cursor()
        for r_i, row in enumerate(tqdm(bt_table.read_rows())):
            row_key = row.row_key

            if row_key == b'table_state':
              continue

            sgf_file = row.cell_value(METADATA, b'sgf').decode()
            timestamp = sgf_file.split('-')[0]
            pb = row.cell_value(METADATA, b'black').decode()
            pw = row.cell_value(METADATA, b'white').decode()
            result = row.cell_value(METADATA, b'result').decode()
            black_won = result.lower().startswith('b')

            assert pw and pb and result, row_key

            # TODO remove before commiting.
            if sgf_file.startswith('v'):
                continue

            # TODO(sethtroisi): At somepoint it would be nice to store this
            # during evaluation and backfill cbt.

            # NOTE: model_names (000123-brave) may be duplicated between runs
            # see 000588-anchorite in v9 and v10.

            if '-v7-' in sgf_file or '-v9-' in sgf_file:
                # very old
                status['old runs'] += 1
                continue

            status['considered'] += 1

            test = determine_model_id(sgf_file, pb, pw, model_runs)
            if test is None:
                status['determine failed'] += 1
                continue

            run_b, run_w = test
            b_model_id = model_ids[(run_b, pb)]
            w_model_id = model_ids[(run_w, pw)]

            """
        new_games = []
        new_wins = []
        model_count_updates = defaultdict(lambda: [0,0,0,0])
            b_model_id = model_ids[(run_b, pb)]
            w_model_id = model_ids[(run_w, pw)]

            new_games.append([
                timestamp, sgf_file,
                b_model_id, w_model_id,
                black_won, result
            ])


            model_count_updates[b_model_id][0] += 1
            model_count_updates[b_model_id][1] += black_won
            model_count_updates[w_model_id]    += [0, 0,       , 1, not black_won]
                """

            try:
                b_id = b_model_id
                w_id = w_model_id

                game_id = None
                try:
                    c.execute("""insert into games(timestamp, filename, b_id, w_id, black_won, result)
                                    values(?, ?, ?, ?, ?, ?)
                    """, [timestamp, sgf_file, b_id, w_id, result.lower().startswith('b'), result])
                    game_id = c.lastrowid
                except sqlite3.IntegrityError:
                    # print("Duplicate game: {}".format(sgf_file))
                    continue

                if game_id is None:
                    print("Somehow, game_id was None")

                # update wins/game counts on model, and wins table.
                c.execute("update models set num_games = num_games + 1 where id in (?, ?)", [b_id, w_id])
                if result.lower().startswith('b'):
                    c.execute("update models set black_games = black_games + 1, black_wins = black_wins + 1 where id = ?", (b_id,))
                    c.execute("update models set white_games = white_games + 1 where id = ?", (w_id,))
                    c.execute("insert into wins(game_id, model_winner, model_loser) values(?, ?, ?)",
                              [game_id, b_id, w_id])
                elif result.lower().startswith('w'):
                    c.execute("update models set black_games = black_games + 1 where id = ?", (b_id,))
                    c.execute("update models set white_games = white_games + 1, white_wins = white_wins + 1 where id = ?", (w_id,))
                    c.execute("insert into wins(game_id, model_winner, model_loser) values(?, ?, ?)",
                              [game_id, w_id, b_id])
                new_games += 1
                if new_games % 1000 == 0:
                    print("committing", new_games)
                    db.commit()
            except:
                print("Bailed!")
                db.rollback()
                raise

    print()
    print("Added {} new games to database".format(new_games))
    print()
    for s, count in status.most_common():
        print("{:<10}".format(count),s )


def compute_ratings(data=None):
    """ Returns the tuples of (model_id, rating, sigma)
    N.B. that `model_id` here is NOT the model number in the run

    'data' is tuples of (winner, loser) model_ids (not model numbers)
    """
    if data is None:
        with sqlite3.connect("cbt_ratings.db") as db:
            data = db.execute("select model_winner, model_loser from wins").fetchall()
    model_ids = set([d[0] for d in data]).union(set([d[1] for d in data]))

    # Map model_ids to a contiguous range.
    ordered = sorted(model_ids)
    new_id = {}
    for i, m in enumerate(ordered):
        new_id[m] = i

    # A function to rewrite the model_ids in our pairs
    def ilsr_data(d):
        p1, p2 = d
        p1 = new_id[p1]
        p2 = new_id[p2]
        return (p1, p2)

    pairs = list(map(ilsr_data, data))
    ilsr_param = choix.ilsr_pairwise(
        len(ordered),
        pairs,
        alpha=0.0001,
        max_iter=800)

    hessian = choix.opt.PairwiseFcts(pairs, penalty=.1).hessian(ilsr_param)
    std_err = np.sqrt(np.diagonal(np.linalg.inv(hessian)))

    # Elo conversion
    elo_mult = 400 / math.log(10)

    min_rating = min(ilsr_param)
    ratings = {}

    for model_id, param, err in zip(ordered, ilsr_param, std_err):
        ratings[model_id] = (elo_mult * (param - min_rating), elo_mult * err)

    return ratings


def top_n(n=10):
    data = wins_subset(fsdb.models_dir())
    r = compute_ratings(data)
    return [(model_num_for(k), v) for v, k in
            sorted([(v, k) for k, v in r.items()])[-n:][::-1]]


def wins_subset(bucket):
    with sqlite3.connect('cbt_ratings.db') as db:
        data = db.execute(
            "select model_winner, model_loser from wins "
            "join models where "
            "    models.bucket = ? AND "
            "    model_winner = models.id",
            (bucket,)).fetchall()
    return data


def main():
    # bt_table =  bigtable.Client("tensor-go",  admin=True).instance("minigo-instance").table("eval_games")
    models_table = (bigtable
                .Client(FLAGS.cbt_project, read_only=True)
                .instance(FLAGS.cbt_instance)
                .table("models"))
    # TODO(djk): exists without admin=True, read_only=False

    eval_games_table = (bigtable
                .Client(FLAGS.cbt_project, read_only=True)
                .instance(FLAGS.cbt_instance)
                .table("eval_games"))

    model_ids, model_runs = setup_models(models_table)

    sync(eval_games_table, model_ids, model_runs)
    return

    data = wins_subset(fsdb.models_dir())
    print(len(data))
    r = compute_ratings(data)
    for v, k in sorted([(v, k) for k, v in r.items()])[-20:][::-1]:
        print(models[model_num_for(k)][1], v)
    db = sqlite3.connect("ratings.db")
    print(db.execute("select count(*) from wins").fetchone()[0], "games")
    for m in models[-10:]:
        m_id = model_id(m[0])
        if m_id in r:
            rat, sigma = r[m_id]
            print("{:>30}:  {:.2f} ({:.3f})".format(m[1], rat, sigma))
        else:
            print("{}, Model id not found({})".format(m[1], m_id))


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    main()
