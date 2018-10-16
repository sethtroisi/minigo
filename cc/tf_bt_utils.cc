// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cc/tf_utils.h"

#include <cstdint>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/mcts_player.h"
#include "google/cloud/bigtable/data_client.h"
#include "google/cloud/bigtable/read_modify_write_rule.h"
#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"

using google::cloud::bigtable::bigendian64_t;
using google::cloud::bigtable::CreateDefaultDataClient;
using google::cloud::bigtable::ReadModifyWriteRule;
using google::cloud::bigtable::SetCell;
using google::cloud::bigtable::Table;

using tensorflow::io::RecordReaderOptions;
using tensorflow::io::SequentialRecordReader;

namespace minigo {
namespace tf_utils {

// Fetches the tensorflow Examples from a MctsPlayer.
// Linked from tf_utils.cc
std::vector<tensorflow::Example> MakeExamples(const MctsPlayer& player);

// Writes a list of tensorflow Example protos to a series of Bigtable rows.
void WriteTfExamples(Table& table, const std::string& row_prefix,
                     const std::vector<tensorflow::Example>& examples) {
  int move = 0;
  for (const auto& example : examples) {
    std::string data;
    example.SerializeToString(&data);
    auto row_name = absl::StrFormat("%s_m_%06d", row_prefix, move);
    google::cloud::bigtable::SingleRowMutation mutation(row_name);
    mutation.emplace_back(SetCell("tfexample", "example", data));
    mutation.emplace_back(SetCell("metadata", "move", std::to_string(move)));
    table.Apply(std::move(mutation));
    move++;
  }
}

void WriteGameExamples(const std::string& gcp_project_name,
                       const std::string& instance_name,
                       const std::string& table_name,
                       const MctsPlayer& player) {
  auto examples = MakeExamples(player);
  Table table(CreateDefaultDataClient(gcp_project_name, instance_name,
                                      google::cloud::bigtable::ClientOptions()),
              table_name);
  // This will be everything from a single game, so retrieve the game
  // counter from the Bigtable and increment it atomically.
  using namespace google::cloud::bigtable;
  auto rule =
      ReadModifyWriteRule::IncrementAmount("metadata", "game_counter", 1);
  auto row = table.ReadModifyWriteRow("table_state", rule);
  uint64_t game_counter = 0;
  for (auto const& cell : row.cells()) {
    if (cell.family_name() == "metadata" &&
        cell.column_qualifier() == "game_counter") {
      game_counter = cell.value_as<bigendian64_t>().get();
      break;
    }
  }

  auto row_prefix = absl::StrFormat("g_%010d", game_counter);
  WriteTfExamples(table, row_prefix, examples);
  std::cerr << "Bigtable rows written to prefix " << row_prefix << " : "
            << examples.size() << std::endl;
}

uint64_t IncrementGameCounter(const std::string& gcp_project_name,
                              const std::string& instance_name,
                              const std::string& table_name, size_t delta) {
  Table table(CreateDefaultDataClient(gcp_project_name, instance_name,
                                      google::cloud::bigtable::ClientOptions()),
              table_name);
  using namespace google::cloud::bigtable;
  auto rule =
      ReadModifyWriteRule::IncrementAmount("metadata", "game_counter", delta);
  auto row = table.ReadModifyWriteRow("table_state", rule);
  for (auto const& cell : row.cells()) {
    if (cell.family_name() == "metadata" &&
        cell.column_qualifier() == "game_counter") {
      return cell.value_as<bigendian64_t>().get();
    }
  }
  MG_FATAL() << "Failed to increment table_state=metadata:game_counter";
  return 0;
}

void PortGamesToBigtable(const std::string& gcp_project_name,
                         const std::string& instance_name,
                         const std::string& table_name,
                         const std::vector<std::string>& paths,
                         int64_t game_counter) {
  auto client_options = google::cloud::bigtable::ClientOptions();
  auto channel_args = client_options.channel_arguments();
  channel_args.SetUserAgentPrefix("minigo_to_cbt");
  channel_args.SetInt(GRPC_ARG_ENABLE_CENSUS, 0);
  channel_args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 0);
  channel_args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 60 * 1000);
  client_options.set_channel_arguments(channel_args);
  Table table(
      CreateDefaultDataClient(gcp_project_name, instance_name, client_options),
      table_name);

  if (game_counter < 0) {
    if (paths.size() != 1) {
      MG_FATAL() << "Atomic game updates require batch size of 1 game";
      return;
    }
    MG_FATAL() << "Have not yet implemented atomic game counter update.";
    return;
  }

  google::cloud::bigtable::BulkMutation game_batch;
  auto start_time = absl::Now();
  int changes = 0;
  for (const auto& path : paths) {
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(path, &file));

    RecordReaderOptions options;
    options.compression_type = RecordReaderOptions::ZLIB_COMPRESSION;
    SequentialRecordReader reader(file.get(), options);

    auto row_prefix = absl::StrFormat("g_%010d", game_counter);

    // Transforms something like:
    //     gs://minigo/data/play/2018-10-14-13/1539522000-8x7lb.tfrecord.zz
    // into:
    //     2018-10-14-13-1539522000-8x7lb
    auto game_id = path;
    game_id.erase(game_id.rfind(".tfrecord.zz"));
    game_id[game_id.rfind('/')] = '-';
    game_id.erase(0, game_id.rfind('/') + 1);

    auto zero_row = absl::StrFormat("%s_m_%06d", row_prefix, 0);
    google::cloud::bigtable::SingleRowMutation zero_row_mutation(zero_row);
    zero_row_mutation.emplace_back(SetCell("metadata", "game_id", game_id));
    game_batch.emplace_back(std::move(zero_row_mutation));

    std::string data;
    int move_number = 0;
    while (reader.ReadRecord(&data).ok()) {
      auto row_name = absl::StrFormat("%s_m_%06d", row_prefix, move_number);
      google::cloud::bigtable::SingleRowMutation row_mutation(row_name);
      row_mutation.emplace_back(SetCell("tfexample", "example", data));
      row_mutation.emplace_back(
          SetCell("metadata", "move", std::to_string(move_number)));
      game_batch.emplace_back(std::move(row_mutation));
      move_number++;
      changes++;
    }
    game_counter++;
  }

  table.BulkApply(std::move(game_batch));
  auto finish_time = absl::Now();
  double elapsed = absl::ToDoubleSeconds(finish_time - start_time);
  VLOG(2) << "Total changes: " << changes << " at " << changes / elapsed
          << " changes/second";
  VLOG(2) << "Total games: " << paths.size() << " at " << paths.size() / elapsed
          << " games/second";
}

}  // namespace tf_utils
}  // namespace minigo
