#!/bin/sh
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

# Added to the player image.
# Wraps our call to main.py

set -e

echo creds: $GOOGLE_APPLICATION_CREDENTIALS
echo bucket: $BUCKET_NAME
echo board_size: $BOARD_SIZE

python3 rl_loop.py selfplay \
  --bucket_name=$BUCKET_NAME \
  --resign-threshold=0.88 \
  --num_readouts=900

echo Finished a set of games!
