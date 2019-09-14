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

#include "cc/model/reloading_model.h"

#include "cc/file/path.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

TEST(ReloadingModelTest, ParseModelPathPattern_Valid) {
  struct Test {
    std::string pattern;
    std::string expected_directory;
    std::string expected_basename_pattern;
  };

  std::vector<Test> tests = {
      {"foo/%d-bar.pb", "foo", "%d-bar.pb"},
      {"foo/bar/%d-bar.pb", "foo/bar", "%d-bar.pb"},
      {"foo/bar/%d", "foo/bar", "%d"},
  };

  for (const auto& test : tests) {
    std::string actual_directory, actual_basename_pattern;
    ASSERT_TRUE(ReloadingModelUpdater::ParseModelPathPattern(
        test.pattern, &actual_directory, &actual_basename_pattern));
    EXPECT_EQ(test.expected_directory, actual_directory);
    EXPECT_EQ(test.expected_basename_pattern, actual_basename_pattern);
  }
}

TEST(ReloadingModelTest, ParseModelPathPattern_Invalid) {
  std::vector<std::string> tests = {
      "nodir.pb", "foo/%x.pb", "%d/foo.pb", "%d", "", "foo/%d-%d",
  };

  for (const auto& pattern : tests) {
    std::string directory, basename_pattern;
    EXPECT_FALSE(ReloadingModelUpdater::ParseModelPathPattern(
        pattern, &directory, &basename_pattern))
        << pattern;
  }
}

TEST(ReloadingModelTest, MatchBasename_Valid) {
  struct Test {
    std::string basename;
    std::string pattern;
    int expected_generation;
  };
  std::vector<Test> tests = {
      {"7", "%d%n", 7},
      {"-2", "%d%n", -2},
      {"-3", "-%d%n", 3},
      {"a-4", "a%d%n", -4},
      {"a-5", "a-%d%n", 5},
      {"a-0042.pb", "a-%d.pb%n", 42},
      {"a-888.pb", "a-%d.pb%n", 888},
      {"xxx0.pb", "xxx%d.pb%n", 0},
  };

  for (const auto& test : tests) {
    int actual_generation;
    ASSERT_TRUE(ReloadingModelUpdater::MatchBasename(
        test.basename, test.pattern, &actual_generation));
    EXPECT_EQ(test.expected_generation, actual_generation);
  }
}

TEST(ReloadingModelTest, MatchBasename_Invalid) {
  struct Test {
    std::string basename;
    std::string pattern;
  };
  std::vector<Test> tests = {
      {"1", "-%d%n"}, {"foo-123.pb", "foo-%d%n"}, {"foo-123", "foo-%d.pb%n"},
      {"%d", "%d%n"}, {"foo", "foo%d%n"},         {"xx1", "foo%d%n"},
  };

  for (const auto& test : tests) {
    int generation;
    EXPECT_FALSE(ReloadingModelUpdater::MatchBasename(
        test.basename, test.pattern, &generation));
  }
}

}  // namespace
}  // namespace minigo
