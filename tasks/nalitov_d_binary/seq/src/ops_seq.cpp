#include "nalitov_d_binary/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>

#include "nalitov_d_binary/common/include/common.hpp"

namespace nalitov_d_binary {

namespace {

constexpr uint8_t kThreshold = 128;

[[nodiscard]] size_t ToIndex(int x, int y, int width) {
  return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
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
  const int total_pixels = width * height;

  std::vector<bool> visited(static_cast<size_t>(total_pixels), false);
  working_image_.components.clear();

  const std::array<std::pair<int, int>, 4> kDirections = {std::make_pair(1, 0), std::make_pair(-1, 0),
                                                          std::make_pair(0, 1), std::make_pair(0, -1)};

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = ToIndex(x, y, width);
      if (working_image_.pixels[idx] == 0 || visited[idx]) {
        continue;
      }

      std::queue<GridPoint> frontier;
      std::vector<GridPoint> component;
      frontier.emplace(x, y);
      visited[idx] = true;

      while (!frontier.empty()) {
        const GridPoint current = frontier.front();
        frontier.pop();
        component.push_back(current);

        for (const auto &[dx, dy] : kDirections) {
          const int nx = current.x + dx;
          const int ny = current.y + dy;

          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
          }

          const size_t nidx = ToIndex(nx, ny, width);
          if (visited[nidx] || working_image_.pixels[nidx] == 0) {
            continue;
          }

          visited[nidx] = true;
          frontier.emplace(nx, ny);
        }
      }

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
  std::sort(sorted_points.begin(), sorted_points.end(), [](const GridPoint &lhs, const GridPoint &rhs) {
    if (lhs.x != rhs.x) {
      return lhs.x < rhs.x;
    }
    return lhs.y < rhs.y;
  });

  sorted_points.erase(std::unique(sorted_points.begin(), sorted_points.end()), sorted_points.end());

  if (sorted_points.size() <= 2U) {
    return sorted_points;
  }

  const auto cross = [](const GridPoint &a, const GridPoint &b, const GridPoint &c) {
    const long long abx = static_cast<long long>(b.x) - static_cast<long long>(a.x);
    const long long aby = static_cast<long long>(b.y) - static_cast<long long>(a.y);
    const long long bcx = static_cast<long long>(c.x) - static_cast<long long>(b.x);
    const long long bcy = static_cast<long long>(c.y) - static_cast<long long>(b.y);
    return (abx * bcy) - (aby * bcx);
  };

  std::vector<GridPoint> lower;
  std::vector<GridPoint> upper;
  lower.reserve(sorted_points.size());
  upper.reserve(sorted_points.size());

  for (const auto &pt : sorted_points) {
    while (lower.size() >= 2U && cross(lower[lower.size() - 2U], lower.back(), pt) <= 0) {
      lower.pop_back();
    }
    lower.push_back(pt);
  }

  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (upper.size() >= 2U && cross(upper[upper.size() - 2U], upper.back(), *it) <= 0) {
      upper.pop_back();
    }
    upper.push_back(*it);
  }

  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

}  // namespace nalitov_d_binary
