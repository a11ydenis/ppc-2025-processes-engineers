#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nalitov_d_binary {

struct GridPoint {
  int x{0};
  int y{0};

  GridPoint() = default;
  GridPoint(int px, int py) : x(px), y(py) {}

  bool operator==(const GridPoint &other) const {
    return x == other.x && y == other.y;
  }

  bool operator!=(const GridPoint &other) const {
    return !(*this == other);
  }

  bool operator<(const GridPoint &other) const {
    return (y < other.y) || (y == other.y && x < other.x);
  }
};

struct BinaryImage {
  int width{0};
  int height{0};
  std::vector<uint8_t> pixels;
  std::vector<std::vector<GridPoint>> components;
  std::vector<std::vector<GridPoint>> convex_hulls;
};

using InType = BinaryImage;
using OutType = BinaryImage;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nalitov_d_binary
