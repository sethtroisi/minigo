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

#include "cc/mcts_player.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "cc/logging.h"
#include "cc/random.h"

namespace minigo {

std::ostream& operator<<(std::ostream& os, const MctsPlayer::Options& options) {
  os << " inject_noise:" << options.inject_noise
     << " soft_pick:" << options.soft_pick
     << " value_init_penalty:" << options.value_init_penalty
     << " policy_softmax_temp:" << options.policy_softmax_temp
     << " virtual_losses:" << options.virtual_losses
     << " num_readouts:" << options.num_readouts
     << " seconds_per_move:" << options.seconds_per_move
     << " time_limit:" << options.time_limit
     << " decay_factor:" << options.decay_factor
     << " fastplay_frequency:" << options.fastplay_frequency
     << " fastplay_readouts:" << options.fastplay_readouts
     << " target_pruning:" << options.target_pruning
     << " random_seed:" << options.random_seed << std::flush;
  return os;
}

float TimeRecommendation(int move_num, float seconds_per_move, float time_limit,
                         float decay_factor) {
  // Divide by two since you only play half the moves in a game.
  int player_move_num = move_num / 2;

  // Sum of geometric series maxes out at endgame_time seconds.
  float endgame_time = seconds_per_move / (1.0f - decay_factor);

  float base_time;
  int core_moves;
  if (endgame_time > time_limit) {
    // There is so little main time that we're already in 'endgame' mode.
    base_time = time_limit * (1.0f - decay_factor);
    core_moves = 0;
  } else {
    // Leave over endgame_time seconds for the end, and play at
    // seconds_per_move for as long as possible.
    base_time = seconds_per_move;
    core_moves = (time_limit - endgame_time) / seconds_per_move;
  }

  return base_time *
         std::pow(decay_factor, std::max(player_move_num - core_moves, 0));
}

MctsPlayer::MctsPlayer(std::unique_ptr<Model> model,
                       std::shared_ptr<InferenceCache> inference_cache,
                       Game* game, const Options& options)
    : model_(std::move(model)),
      game_root_(&root_stats_, {&bv_, &gv_, Color::kBlack}),
      game_(game),
      rnd_(options.random_seed, Random::kUniqueStream),
      options_(options),
      inference_cache_(std::move(inference_cache)),
      inference_mix_(rnd_.UniformUint64()) {
  // When to do deterministic move selection: 30 moves on a 19x19, 6 on 9x9.
  // divide 2, multiply 2 guarentees that white and black do even number.
  temperature_cutoff_ = !options_.soft_pick ? -1 : (((kN * kN / 12) / 2) * 2);
  root_ = &game_root_;

  NewGame();
}

MctsPlayer::~MctsPlayer() = default;

void MctsPlayer::InitializeGame(const Position& position) {
  root_stats_ = {};
  game_root_ = MctsNode(&root_stats_, Position(&bv_, &gv_, position));
  root_ = &game_root_;
  game_->NewGame();
}

void MctsPlayer::NewGame() { InitializeGame({&bv_, &gv_, Color::kBlack}); }

bool MctsPlayer::UndoMove() {
  if (root_ == &game_root_) {
    return false;
  }
  root_ = root_->parent;
  game_->UndoMove();
  if (!options_.tree_reuse) {
    // Reset the root to a fresh node.
    if (root_->parent != nullptr) {
      *root_ = MctsNode(root_->parent, root_->move);
    } else {
      *root_ = MctsNode(root_->stats, root_->position);
    }
  }
  return true;
}

Coord MctsPlayer::SuggestMove(int new_readouts, bool inject_noise) {
  auto start = absl::Now();

  // In order to correctly count the number of reads performed, the root node
  // must be expanded. The root will always be expanded unless this is the first
  // time SuggestMove has been called for a game, or PlayMove was called without
  // a prior call to SuggestMove.
  if (!root_->HasFlag(MctsNode::Flag::kExpanded)) {
    TreeSearch(1);
  }

  if (inject_noise) {
    std::array<float, kNumMoves> noise;
    rnd_.Dirichlet(kDirichletAlpha, &noise);
    root_->InjectNoise(noise, options_.noise_mix);
  }
  int current_readouts = root_->N();

  if (options_.seconds_per_move > 0) {
    // Use time to limit the number of reads.
    float seconds_per_move = options_.seconds_per_move;
    if (options_.time_limit > 0) {
      seconds_per_move =
          TimeRecommendation(root_->position.n(), seconds_per_move,
                             options_.time_limit, options_.decay_factor);
    }
    while (absl::Now() - start < absl::Seconds(seconds_per_move)) {
      TreeSearch(options_.virtual_losses);
    }
  } else {
    // Use a fixed number of reads.
    while (root_->N() < current_readouts + new_readouts) {
      TreeSearch(options_.virtual_losses);
    }
  }
  if (ShouldResign()) {
    return Coord::kResign;
  }

  // Pick the move before altering the tree for training targets.
  auto c = PickMove();

  // After picking the move, destructively adjust the visit counts
  // according to whatever flag-controlled scheme.
  if (options_.target_pruning && inject_noise) {
    root_->ReshapeFinalVisits();
  }

  return c;
}

Coord MctsPlayer::PickMove() {
  if (root_->position.n() >= temperature_cutoff_) {
    return root_->GetMostVisitedMove();
  }

  // Select from the first kN * kN moves (instead of kNumMoves) to avoid
  // randomly choosing to pass early on in the game.
  std::array<float, kN * kN> cdf;

  // For moves before the temperature cutoff, exponentiate the probabilities by
  // a temperature slightly larger than unity to encourage diversity in early
  // play and hopefully to move away from 3-3s.
  for (size_t i = 0; i < cdf.size(); ++i) {
    cdf[i] = std::pow(root_->child_N(i), options_.policy_softmax_temp);
  }
  for (size_t i = 1; i < cdf.size(); ++i) {
    cdf[i] += cdf[i - 1];
  }

  if (cdf.back() == 0) {
    // It's actually possible for an early model to put all its reads into pass,
    // in which case the SearchSorted call below will always return 0. In this
    // case, we'll just let the model have its way and allow a pass.
    return Coord::kPass;
  }

  Coord c = rnd_.SampleCdf(absl::MakeSpan(cdf));
  MG_DCHECK(root_->child_N(c) != 0);
  return c;
}

void MctsPlayer::TreeSearch(int num_leaves) {
  SelectLeaves(num_leaves);
  ProcessLeaves();
}

void MctsPlayer::SelectLeaves(int num_leaves) {
  tree_search_inferences_.clear();
  Model::Output cached_output;

  int max_cache_misses = num_leaves * 2;
  int num_selected = 0;
  int num_cache_misses = 0;
  int num_cache_hits = 0;
  while (num_cache_misses < max_cache_misses) {
    auto* leaf = root_->SelectLeaf();

    if (leaf->game_over() || leaf->at_move_limit()) {
      float value =
          leaf->position.CalculateScore(game_->options().komi) > 0 ? 1 : -1;
      leaf->IncorporateEndGameResult(value, root_);
      ++num_cache_misses;
      continue;
    }

    // Calculate the symmetry we want to use for inference.
    auto inference_sym = GetInferenceSymmetry(leaf);
    auto canonical_sym = GetCanonicalSymmetry(leaf);

    InferenceCache::Key cache_key;
    if (inference_cache_ != nullptr) {
      cache_key =
          InferenceCache::Key(leaf->move, canonical_sym, leaf->position);

      if (inference_cache_->TryGet(cache_key, canonical_sym, inference_sym,
                                   &cached_output)) {
        ++num_cache_hits;
        leaf->IncorporateResults(options_.value_init_penalty,
                                 cached_output.policy, cached_output.value,
                                 root_);
        continue;
      }
    }

    ++num_cache_misses;

    tree_search_inferences_.emplace_back(cache_key, canonical_sym,
                                         inference_sym, leaf);

    auto& input = tree_search_inferences_.back().input;
    input.to_play = leaf->position.to_play();
    input.sym = inference_sym;
    // TODO(tommadams): add a method to Model that returns the required position
    // history size.
    auto* node = leaf;
    for (int i = 0; i < input.position_history.capacity(); ++i) {
      input.position_history.push_back(&node->position.stones());
      node = node->parent;
      if (node == nullptr) {
        break;
      }
    }

    leaf->AddVirtualLoss(root_);
    if (++num_selected == num_leaves) {
      // We found enough leaves.
      break;
    }
    if (leaf == root_) {
      // If the root is a leaf, we can't possibly find any other leaves.
      break;
    }
  }
}

bool MctsPlayer::ShouldResign() const {
  return game_->options().resign_enabled &&
         root_->Q_perspective() < game_->options().resign_threshold;
}

void MctsPlayer::SetTreeSearchCallback(TreeSearchCallback cb) {
  tree_search_cb_ = std::move(cb);
}

std::string MctsPlayer::GetModelsUsedForInference() const {
  std::vector<std::string> parts;
  parts.reserve(inferences_.size());
  for (const auto& info : inferences_) {
    parts.push_back(absl::StrCat(info.model, "(", info.first_move, ",",
                                 info.last_move, ")"));
  }
  return absl::StrJoin(parts, ", ");
}

bool MctsPlayer::PlayMove(Coord c) {
  if (root_->game_over()) {
    MG_LOG(ERROR) << "Can't play move " << c << ", game is over";
    return false;
  }

  // Handle resignations.
  if (c == Coord::kResign) {
    game_->SetGameOverBecauseOfResign(OtherColor(root_->position.to_play()));
    return true;
  }

  if (!root_->position.legal_move(c)) {
    MG_LOG(ERROR) << "Move " << c << " is illegal";
    // We're probably about to crash. Dump the player's options and the moves
    // that got us to this point.
    MG_LOG(ERROR) << "MctsPlayer options: " << options_;
    MG_LOG(ERROR) << "Game options: " << game_->options();
    for (int i = 0; i < game_->num_moves(); ++i) {
      const auto* move = game_->GetMove(i);
      MG_LOG(ERROR) << move->color << "  " << move->c;
    }
    return false;
  }

  UpdateGame(c);

  if (options_.tree_reuse) {
    root_ = root_->MaybeAddChild(c);
    // TODO(tommadams): remove the prune_orphaned_nodes and always prune
    // children once the inference cache is fully working. Disable child pruning
    // was a temporary hack to make Minigui usable.
    if (options_.prune_orphaned_nodes) {
      // Don't need to keep the parent's children around anymore because we'll
      // never revisit them during normal play.
      root_->parent->PruneChildren(c);
    }
  } else {
    root_->children.clear();
    root_ = root_->MaybeAddChild(c);
  }

  // Handle consecutive passing or termination by move limit.
  if (root_->at_move_limit()) {
    game_->SetGameOverBecauseMoveLimitReached(
        root_->position.CalculateScore(game_->options().komi));
  } else if (root_->game_over()) {
    game_->SetGameOverBecauseOfPasses(
        root_->position.CalculateScore(game_->options().komi));
  }

  return true;
}

void MctsPlayer::UpdateGame(Coord c) {
  // Record which model(s) were used when running tree search for this move.
  std::vector<std::string> models;
  if (!inferences_.empty()) {
    for (auto it = inferences_.rbegin(); it != inferences_.rend(); ++it) {
      if (it->last_move < root_->position.n()) {
        break;
      }
      models.push_back(it->model);
    }
    std::reverse(models.begin(), models.end());
  }

  // Build a comment for the move.
  auto comment = root_->Describe();
  if (!models.empty()) {
    comment =
        absl::StrCat("models:", absl::StrJoin(models, ","), "\n", comment);
  }

  // Convert child visit counts to a probability distribution, pi.
  std::array<float, kNumMoves> search_pi;
  if (root_->position.n() < temperature_cutoff_) {
    // Squash counts before normalizing to match softpick behavior in PickMove.
    for (int i = 0; i < kNumMoves; ++i) {
      search_pi[i] = std::pow(root_->child_N(i), options_.policy_softmax_temp);
    }
  } else {
    for (int i = 0; i < kNumMoves; ++i) {
      search_pi[i] = root_->child_N(i);
    }
  }
  // Normalize counts.
  float sum = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum += search_pi[i];
  }
  for (int i = 0; i < kNumMoves; ++i) {
    search_pi[i] /= sum;
  }

  // Update the game history.
  game_->AddMove(root_->position.to_play(), c, root_->position.stones(),
                 std::move(comment), root_->Q(), search_pi, std::move(models));
}

// TODO(tommadams): move this up to below SelectLeaves.
void MctsPlayer::ProcessLeaves() {
  if (tree_search_inferences_.empty()) {
    return;
  }

  input_ptrs_.clear();
  output_ptrs_.clear();
  for (auto& x : tree_search_inferences_) {
    input_ptrs_.push_back(&x.input);
    output_ptrs_.push_back(&x.output);
  }

  // Run inference.
  model_->RunMany(input_ptrs_, &output_ptrs_, &inference_model_);

  // Record some information about the inference.
  if (!inference_model_.empty()) {
    if (inferences_.empty() || inference_model_ != inferences_.back().model) {
      inferences_.emplace_back(inference_model_, root_->position.n());
    }
    inferences_.back().last_move = root_->position.n();
    inferences_.back().total_count += tree_search_inferences_.size();
  }

  // Incorporate the inference outputs back into tree search.
  for (auto& inference : tree_search_inferences_) {
    auto& output = inference.output;

    // Merge the inference output with those in the inference cache, possibly
    // updating the values in `output`.
    if (inference_cache_ != nullptr) {
      inference_cache_->Merge(inference.cache_key, inference.canonical_sym,
                              inference.inference_sym, &output);
    }

    // Propagate the results back up the tree to the root.
    auto* leaf = inference.leaf;
    leaf->IncorporateResults(options_.value_init_penalty, output.policy,
                             output.value, root_);
    leaf->RevertVirtualLoss(root_);
  }

  if (tree_search_cb_ != nullptr) {
    std::vector<const MctsNode*> leaves;
    leaves.reserve(tree_search_inferences_.size());
    for (auto& inference : tree_search_inferences_) {
      leaves.push_back(inference.leaf);
    }
    tree_search_cb_(leaves);
  }
}

}  // namespace minigo
