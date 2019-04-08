from google.cloud import bigtable
from google.cloud.bigtable import row_filters
from tqdm import tqdm
import re

bt_table =  bigtable.Client("tensor-go",  admin=True).instance("minigo-instance").table("eval_games")

BAD_NAME = re.compile("[a-z][0-9]{6}-[a-z-]*$")
count = 0

rows = bt_table.read_rows(
  filter_=row_filters.ColumnRangeFilter('metadata', b'sgf', b'sgf')
)

for row in tqdm(rows):
  if row.row_key == b'table_state': continue

  sgf = row.cell_value('metadata', b'sgf').decode()

  match = BAD_NAME.search(sgf)
  if match:
    count += 1
    print(count, row.row_key, sgf)
    update = bt_table.row(row.row_key)
 #   update.delete()
 #   update.commit()
