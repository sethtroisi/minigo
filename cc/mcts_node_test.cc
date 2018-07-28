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

#include "cc/mcts_node.h"

#include <array>
#include <set>

#include "cc/position.h"
#include "cc/random.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

static constexpr char kAlmostDoneBoard[] = R"(
    .XO.XO.OO
    X.XXOOOO.
    XXXXXOOOO
    XXXXXOOOO
    .XXXXOOO.
    XXXXXOOOO
    .XXXXOOO.
    XXXXXOOOO
    XXXXOOOOO)";

// Verifies that no matter who is to play, when we know nothing else, the priors
// should be respected, and the same move should be picked.
TEST(MctsNodeTest, ActionFlipping) {
  Random rnd(1);

  std::array<float, kNumMoves> probs;
  std::uniform_real_distribution<float> dist(0.02, 0.021);
  for (float& prob : probs) {
    prob = rnd();
  }

  MctsNode::EdgeStats black_stats, white_stats;
  MctsNode black_root(&black_stats, TestablePosition("", Color::kBlack));
  MctsNode white_root(&white_stats, TestablePosition("", Color::kWhite));

  black_root.SelectLeaf()->IncorporateResults(probs, 0, &black_root);
  white_root.SelectLeaf()->IncorporateResults(probs, 0, &white_root);
  auto* black_leaf = black_root.SelectLeaf();
  auto* white_leaf = white_root.SelectLeaf();
  EXPECT_EQ(black_leaf->move, white_leaf->move);
  EXPECT_EQ(black_root.CalculateChildActionScore(),
            white_root.CalculateChildActionScore());
}

// Verfies that SelectLeaf chooses the child with the highest action score.
TEST(MctsNodeTest, SelectLeaf) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }
  Coord c = Coord::FromKgs("D9");
  probs[c] = 0.4;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);

  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  EXPECT_EQ(root.position.to_play(), Color::kWhite);
  auto* leaf = root.SelectLeaf();
  EXPECT_EQ(leaf, root.children[c].get());
}

// Verifies IncorporateResults and BackupValue.
TEST(MctsNodeTest, BackupIncorporateResults) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  auto* leaf = root.SelectLeaf();
  leaf->IncorporateResults(probs, -1, &root);  // white wins!

  // Root was visited twice: first at the root, then at this child.
  EXPECT_EQ(root.N(), 2);
  // Root has 0 as a prior and two visits with value 0, -1.
  EXPECT_FLOAT_EQ(root.Q(), -1.0 / 3);  // average of 0, 0, -1
  // Leaf should have one visit
  EXPECT_EQ(root.child_N(leaf->move), 1);
  EXPECT_EQ(leaf->N(), 1);
  // And that leaf's value had its parent's Q (0) as a prior, so the Q
  // should now be the average of 0, -1
  EXPECT_FLOAT_EQ(leaf->Q(), -0.5);

  // We're assuming that SelectLeaf() returns a leaf like:
  //   root
  //     |
  //     leaf
  //       |
  //       leaf2
  // which happens in this test because root is W to play and leaf was a W win.
  EXPECT_EQ(root.position.to_play(), Color::kWhite);
  auto* leaf2 = root.SelectLeaf();
  leaf2->IncorporateResults(probs, -0.2, &root);  // another white semi-win
  EXPECT_EQ(root.N(), 3);
  // average of 0, 0, -1, -0.2
  EXPECT_FLOAT_EQ(root.Q(), -0.3);

  EXPECT_EQ(leaf->N(), 2);
  EXPECT_EQ(leaf2->N(), 1);
  // average of 0, -1, -0.2
  EXPECT_FLOAT_EQ(leaf->Q(), -0.4);
  // average of 0, -0.2
  EXPECT_FLOAT_EQ(leaf2->Q(), -0.1);
}

// Verifies Child_U updates from Parent Q.
TEST(MctsNodeTest, ChildUUpdatesFromParentQ) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, -0.5, &root);

  // Select two adjacent leaf nodes.
  auto* leaf1 = root.SelectLeaf();
  leaf1->AddVirtualLoss(&root);
  auto* leaf2 = root.SelectLeaf();
  leaf1->RevertVirtualLoss(&root);

  // Not the same leaf
  EXPECT_NE(leaf1, leaf2);
  EXPECT_EQ(leaf1->parent, leaf1->parent);

  // N is incremented when SelectLeaf is called to find Leaf1 and leaf2.
  EXPECT_EQ(root.N(), 3);
  EXPECT_FLOAT_EQ(root.Q(), -0.5/4);
  EXPECT_FLOAT_EQ(leaf1->Q(), 0);
  EXPECT_FLOAT_EQ(leaf2->Q(), 0);
  // Leaf1 and leaf2 have already incremented N from select_leaf.
  EXPECT_FLOAT_EQ(root.child_Q(leaf1->move), root.Q()/2);
  EXPECT_FLOAT_EQ(root.child_Q(leaf2->move), root.Q()/2);

  leaf1->IncorporateResults(probs, -1, &root);  // white wins!

  // Leaf1 and root incorporate result directly into Q
  EXPECT_FLOAT_EQ(root.Q(), -1.5 / 4);
  EXPECT_FLOAT_EQ(leaf1->Q(), -0.5);
  // Leaf2 is unchanged.
  EXPECT_FLOAT_EQ(leaf2->Q(), 0);

  // Child_Q used for action score is updated for both children.
  EXPECT_FLOAT_EQ(root.child_Q(leaf1->move), (root.Q() + leaf1->W())/2);
  EXPECT_FLOAT_EQ(root.child_Q(leaf2->move), root.Q()/2);
}

TEST(MctsNodeTest, DoNotExplorePastFinish) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  auto* first_pass = root.MaybeAddChild(Coord::kPass);
  first_pass->IncorporateResults(probs, 0, &root);
  auto* second_pass = first_pass->MaybeAddChild(Coord::kPass);
  EXPECT_DEATH(second_pass->IncorporateResults(probs, 0, &root),
               "is_game_over");
  float value = second_pass->position.CalculateScore(0) > 0 ? 1 : -1;
  second_pass->IncorporateEndGameResult(value, &root);
  auto* node_to_explore = second_pass->SelectLeaf();
  // should just stop exploring at the end position.
  EXPECT_EQ(node_to_explore, second_pass);
}

TEST(MctsNodeTest, AddChild) {
  MctsNode::EdgeStats root_stats;
  TestablePosition board("");
  MctsNode root(&root_stats, board);

  Coord c = Coord::FromKgs("B9");
  auto* child = root.MaybeAddChild(c);
  EXPECT_EQ(1, root.children.count(c));
  EXPECT_EQ(child->parent, &root);
  EXPECT_EQ(child->move, c);
}

TEST(MctsNodeTest, AddChildIdempotency) {
  MctsNode::EdgeStats root_stats;
  TestablePosition board("");
  MctsNode root(&root_stats, board);

  Coord c = Coord::FromKgs("B9");
  auto* child = root.MaybeAddChild(c);
  EXPECT_EQ(1, root.children.count(c));
  EXPECT_EQ(1, root.children.size());
  auto* child2 = root.MaybeAddChild(c);
  EXPECT_EQ(child, child2);
  EXPECT_EQ(1, root.children.count(c));
  EXPECT_EQ(1, root.children.size());
}

TEST(MctsNodeTest, NeverSelectIllegalMoves) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }
  // let's say the NN were to accidentally put a high weight on an illegal move
  probs[1] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  // and let's say the root were visited a lot of times, which pumps up the
  // action score for unvisited moves...
  root.stats->N = 100000;
  for (int i = 0; i < kNumMoves; ++i) {
    if (root.position.IsMoveLegal(i)) {
      root.edges[i].N = 10000;
    }
  }
  // this should not throw an error...
  auto* leaf = root.SelectLeaf();
  // the returned leaf should not be the illegal move
  EXPECT_NE(leaf->move, 1);

  // and even after injecting noise, we should still not select an illegal move
  Random rnd(1);
  for (int i = 0; i < 10; ++i) {
    std::array<float, kNumMoves> noise;
    rnd.Uniform(0, 1, &noise);
    root.InjectNoise(noise);
    leaf = root.SelectLeaf();
    EXPECT_NE(leaf->move, 1);
  }
}

TEST(MctsNodeTest, DontPickUnexpandedChild) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  // Make one move really likely so that tree search goes down that path twice
  // even with a virtual loss.
  probs[17] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  auto* leaf1 = root.SelectLeaf();
  EXPECT_EQ(17, leaf1->move);
  leaf1->AddVirtualLoss(&root);

  auto* leaf2 = root.SelectLeaf();
  EXPECT_EQ(leaf2, leaf2);
}

// Verifies that even when one move is hugely more likely than all the others,
// SelectLeaf will eventually start exploring other moves given enough
// iterations.
TEST(MctsNodeTest, TestSelectLeaf) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  probs[17] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  std::set<MctsNode*> leaves;

  auto* leaf = root.SelectLeaf();
  EXPECT_EQ(17, leaf->move);
  leaf->AddVirtualLoss(&root);
  leaves.insert(leaf);

  for (int i = 0; i < 1000; ++i) {
    leaf = root.SelectLeaf();
    leaf->AddVirtualLoss(&root);
    leaves.insert(leaf);
  }

  // We should have selected at least 2 leaves.
  EXPECT_LE(2, leaves.size());
}

TEST(MctsNodeTest, NormalizeTest) {
  // Generate probability with sum of policy less than 1
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  // Five times larger to test normalization
  probs[17] = 0.005;
  probs[18] = 0;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition("");
  MctsNode root(&root_stats, board);
  root.IncorporateResults(probs, 0, &root);

  // Adjust for the one value that is five times larger and one missing value.
  float normalized = 1.0 / (kNumMoves - 1 + 4);
  for (int i = 0; i < kNumMoves; ++i) {
    if (i == 17) {
      EXPECT_FLOAT_EQ(5 * normalized, root.child_P(i));
    } else if (i == 18) {
      EXPECT_FLOAT_EQ(0, root.child_P(i));
    } else {
      EXPECT_FLOAT_EQ(normalized, root.child_P(i));
    }
  }
}

TEST(MctsNodeTest, InjectNoiseOnlyLegalMoves) {
  // Give moves a uniform policy value.
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.IncorporateResults(probs, 0, &root);

  // kAlmostDoneBoard has 6 legal moves including pass.
  float uniform_policy = 1.0 / 6;

  for (int i = 0; i < kNumMoves; ++i) {
    if (root.illegal_moves[i]) {
      EXPECT_FLOAT_EQ(0, root.edges[i].P);
    } else {
      EXPECT_FLOAT_EQ(uniform_policy, root.edges[i].P);
    }
  }

  // and even after injecting noise, we should still not select an illegal move
  Random rnd(1);
  std::array<float, kNumMoves> noise;
  rnd.Uniform(0, 1, &noise);
  root.InjectNoise(noise);

  for (int i = 0; i < kNumMoves; ++i) {
    if (root.illegal_moves[i]) {
      EXPECT_FLOAT_EQ(0, root.edges[i].P);
    } else {
      EXPECT_LT(0.75 * uniform_policy, root.edges[i].P);
      EXPECT_GT(0.75 * uniform_policy + 0.25, root.edges[i].P);
    }
  }
}

}  // namespace
}  // namespace minigo
