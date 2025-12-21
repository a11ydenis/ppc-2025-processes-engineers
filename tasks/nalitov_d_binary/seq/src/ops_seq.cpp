#include "nalitov_d_binary/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <ranges>
#include <utility>
#include <vector>

#include "nalitov_d_binary/common/include/common.hpp"

namespace nalitov_d_binary {

namespace {

constexpr uint8_t kThreshold = 128;

[[nodiscard]] size_t ToIndex(int x, int y, int width) {
  return (static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x);
}

void FloodFillComponent(int start_x, int start_y, int width, int height, const std::vector<uint8_t> &pixels,
                        std::vector<bool> &visited, std::vector<GridPoint> &component) {
  const std::array<std::pair<int, int>, 4> directions = {std::make_pair(1, 0), std::make_pair(-1, 0),
                                                         std::make_pair(0, 1), std::make_pair(0, -1)};

  std::queue<GridPoint> frontier;
  frontier.emplace(start_x, start_y);
  visited[ToIndex(start_x, start_y, width)] = true;

  while (!frontier.empty()) {
    const GridPoint current = frontier.front();
    frontier.pop();
    component.push_back(current);

    for (const auto &[dx, dy] : directions) {
      const int next_x = current.x + dx;
      const int next_y = current.y + dy;

      if (next_x < 0 || next_x >= width || next_y < 0 || next_y >= height) {
        continue;
      }

      const size_t next_idx = ToIndex(next_x, next_y, width);
      if (visited[next_idx] || pixels[next_idx] == 0) {
        continue;
      }

      visited[next_idx] = true;
      frontier.emplace(next_x, next_y);
    }
  }
}

}  // namespace

NalitovDBinarySEQ::NalitovDBinarySEQ(const InType &in) : working_image_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool NalitovDBinarySEQ::ValidationImpl() {
  const auto &input = GetInput();
  const bool valid_dimensions = input.width > 0 && input.height > 0;
  const bool size_matches = input.pixels.size() == static_cast<size_t>(input.width) * static_cast<size_t>(input.height);
  return valid_dimensions && size_matches;
}

bool NalitovDBinarySEQ::PreProcessingImpl() {
  working_image_ = GetInput();
  ThresholdImage();
  return true;
}

bool NalitovDBinarySEQ::RunImpl() {
  DiscoverComponents();

  working_image_.convex_hulls.clear();
  working_image_.convex_hulls.reserve(working_image_.components.size());

  for (const auto &component : working_image_.components) {
    if (component.empty()) {
      continue;
    }

    if (component.size() <= 2U) {
      working_image_.convex_hulls.push_back(component);
    } else {
      working_image_.convex_hulls.push_back(BuildConvexHull(component));
    }
  }

  GetOutput() = working_image_;
  return true;
}

bool NalitovDBinarySEQ::PostProcessingImpl() {
  return true;
}

void NalitovDBinarySEQ::ThresholdImage() {
  for (auto &pixel : working_image_.pixels) {
    pixel = pixel > kThreshold ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
  }
}

void NalitovDBinarySEQ::DiscoverComponents() {
  const int width = working_image_.width;
  const int height = working_image_.height;
  const size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

  std::vector<bool> visited(total_pixels, false);
  working_image_.components.clear();

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const size_t idx = ToIndex(col, row, width);

      if (working_image_.pixels[idx] == 0 || visited[idx]) {
        continue;
      }

      std::vector<GridPoint> component;
      FloodFillComponent(col, row, width, height, working_image_.pixels, visited, component);

      if (!component.empty()) {
        working_image_.components.push_back(std::move(component));
      }
    }
  }
}

std::vector<GridPoint> NalitovDBinarySEQ::BuildConvexHull(const std::vector<GridPoint> &points) {
  if (points.size() <= 2U) {
    return points;
  }

  std::vector<GridPoint> sorted_points = points;

  std::ranges::sort(sorted_points, [](const GridPoint &lhs, const GridPoint &rhs) {
    return (lhs.x != rhs.x) ? (lhs.x < rhs.x) : (lhs.y < rhs.y);
  });

  auto unique_end = std::ranges::unique(sorted_points).begin();
  sorted_points.erase(unique_end, sorted_points.end());

  if (sorted_points.size() <= 2U) {
    return sorted_points;
  }

  const auto cross = [](const GridPoint &a, const GridPoint &b, const GridPoint &c) {
    const std::int64_t abx = static_cast<std::int64_t>(b.x) - static_cast<std::int64_t>(a.x);
    const std::int64_t aby = static_cast<std::int64_t>(b.y) - static_cast<std::int64_t>(a.y);
    const std::int64_t bcx = static_cast<std::int64_t>(c.x) - static_cast<std::int64_t>(b.x);
    const std::int64_t bcy = static_cast<std::int64_t>(c.y) - static_cast<std::int64_t>(b.y);
    return (abx * bcy) - (aby * bcx);
  };

  std::vector<GridPoint> lower;
  std::vector<GridPoint> upper;
  lower.reserve(sorted_points.size());
  upper.reserve(sorted_points.size());

  for (const auto &point : sorted_points) {
    while (lower.size() >= 2U && cross(lower[lower.size() - 2U], lower.back(), point) <= 0) {
      lower.pop_back();
    }
    lower.push_back(point);
  }

  for (const auto &point : std::ranges::reverse_view(sorted_points)) {
    while (upper.size() >= 2U && cross(upper[upper.size() - 2U], upper.back(), point) <= 0) {
      upper.pop_back();
    }
    upper.push_back(point);
  }

  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

}  // namespace nalitov_d_binary
