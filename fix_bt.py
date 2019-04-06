from google.cloud import bigtable
from tqdm import tqdm
import re

bt_table =  bigtable.Client("tensor-go",  admin=True).instance("minigo-instance").table("eval_games")

BAD_NAME = re.compile("v12-754-vs-v10-674-.*-000790-courageux")
count = 0
for row in tqdm(bt_table.read_rows()):
  if row.row_key == b'table_state': continue

  sgf = row.cell_value('metadata', b'sgf').decode()

  match = BAD_NAME.search(sgf)
  if match:
    count += 1
    print(count, row.row_key, sgf)
    update = bt_table.row(row.row_key)
    update.delete()
    update.commit()
