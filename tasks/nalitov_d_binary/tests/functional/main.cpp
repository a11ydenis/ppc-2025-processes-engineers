#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

#include "nalitov_d_binary/common/include/common.hpp"
#include "nalitov_d_binary/mpi/include/ops_mpi.hpp"
#include "nalitov_d_binary/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace nalitov_d_binary {

namespace {

struct Pattern {
  BinaryImage image;
  std::vector<std::vector<GridPoint>> expected_hulls;
};

BinaryImage MakeBlankImage(int width, int height, uint8_t value = 0) {
  BinaryImage img;
  img.width = width;
  img.height = height;
  img.pixels.assign((static_cast<size_t>(width) * static_cast<size_t>(height)), value);
  return img;
}

void SetPixel(BinaryImage &image, int x, int y, uint8_t value) {
  const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(image.width)) + static_cast<size_t>(x);
  image.pixels[idx] = value;
}

Pattern MakeSinglePointPattern() {
  Pattern pattern;
  pattern.image = MakeBlankImage(5, 5);
  SetPixel(pattern.image, 2, 2, 180);
  pattern.expected_hulls = {{{2, 2}}};
  return pattern;
}

Pattern MakeTwoPointPattern() {
  Pattern pattern;
  pattern.image = MakeBlankImage(6, 6);
  SetPixel(pattern.image, 1, 1, 255);
  SetPixel(pattern.image, 4, 4, 200);
  pattern.expected_hulls = {{{1, 1}}, {{4, 4}}};
  return pattern;
}

Pattern MakeHorizontalLinePattern() {
  Pattern pattern;
  pattern.image = MakeBlankImage(7, 3);
  for (int col = 2; col <= 4; ++col) {
    SetPixel(pattern.image, col, 1, 190);
  }
  pattern.expected_hulls = {{{2, 1}, {4, 1}}};
  return pattern;
}

Pattern MakeFilledRectanglePattern() {
  Pattern pattern;
  pattern.image = MakeBlankImage(8, 8);
  for (int row = 2; row <= 5; ++row) {
    for (int col = 3; col <= 6; ++col) {
      SetPixel(pattern.image, col, row, 255);
    }
  }
  pattern.expected_hulls = {{{3, 2}, {6, 2}, {6, 5}, {3, 5}}};
  return pattern;
}

Pattern MakeDiamondPattern() {
  Pattern pattern;
  pattern.image = MakeBlankImage(9, 9);
  for (int row = 0; row < 9; ++row) {
    for (int col = 0; col < 9; ++col) {
      if (std::abs(col - 4) + std::abs(row - 4) <= 4) {
        SetPixel(pattern.image, col, row, 255);
      }
    }
  }
  pattern.expected_hulls = {{{0, 4}, {4, 0}, {8, 4}, {4, 8}}};
  return pattern;
}

const std::vector<Pattern> &GetAllPatterns() {
  static const std::vector<Pattern> kPatterns = []() {
    std::vector<Pattern> result;
    result.push_back(MakeSinglePointPattern());
    result.push_back(MakeTwoPointPattern());
    result.push_back(MakeHorizontalLinePattern());
    result.push_back(MakeFilledRectanglePattern());
    result.push_back(MakeDiamondPattern());
    return result;
  }();
  return kPatterns;
}

const Pattern &GetPattern(int id) {
  const auto &patterns = GetAllPatterns();
  return patterns.at(static_cast<size_t>(id));
}

std::vector<GridPoint> NormaliseHull(std::vector<GridPoint> hull) {
  std::ranges::sort(hull, [](const GridPoint &lhs, const GridPoint &rhs) {
    return (lhs.y != rhs.y) ? (lhs.y < rhs.y) : (lhs.x < rhs.x);
  });

  const auto unique_range = std::ranges::unique(hull);
  hull.erase(unique_range.begin(), hull.end());
  return hull;
}

std::vector<std::vector<GridPoint>> NormaliseHulls(const std::vector<std::vector<GridPoint>> &hulls) {
  std::vector<std::vector<GridPoint>> normalised;
  normalised.reserve(hulls.size());
  for (const auto &hull : hulls) {
    normalised.push_back(NormaliseHull(hull));
  }

  auto comparator = [](const std::vector<GridPoint> &lhs, const std::vector<GridPoint> &rhs) {
    if (lhs.size() != rhs.size()) {
      return lhs.size() < rhs.size();
    }
    return std::ranges::lexicographical_compare(
        lhs, rhs, [](const GridPoint &a, const GridPoint &b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); });
  };

  std::ranges::sort(normalised, comparator);
  return normalised;
}

bool HullsMatch(const std::vector<std::vector<GridPoint>> &expected,
                const std::vector<std::vector<GridPoint>> &actual) {
  auto norm_expected = NormaliseHulls(expected);
  auto norm_actual = NormaliseHulls(actual);
  return norm_expected == norm_actual;
}

}  // namespace

class NalitovDBinaryFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    auto test_param = std::get<2>(GetParam());
    int pattern_id = std::get<0>(test_param);
    const auto &pattern = GetPattern(pattern_id);
    return HullsMatch(pattern.expected_hulls, output_data.convex_hulls);
  }

  InType GetTestInputData() final {
    auto test_param = std::get<2>(GetParam());
    int pattern_id = std::get<0>(test_param);
    return GetPattern(pattern_id).image;
  }
};

namespace {

TEST_P(NalitovDBinaryFuncTests, ConvexHullPatterns) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(0, "single_point"), std::make_tuple(1, "two_points"),
                                            std::make_tuple(2, "horizontal_line"),
                                            std::make_tuple(3, "filled_rectangle"), std::make_tuple(4, "diamond")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<NalitovDBinaryMPI, InType>(kTestParam, PPC_SETTINGS_nalitov_d_binary),
                   ppc::util::AddFuncTask<NalitovDBinarySEQ, InType>(kTestParam, PPC_SETTINGS_nalitov_d_binary));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NalitovDBinaryFuncTests::PrintFuncTestName<NalitovDBinaryFuncTests>;

INSTANTIATE_TEST_SUITE_P(ConvexHullBinaryImage, NalitovDBinaryFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nalitov_d_binary
