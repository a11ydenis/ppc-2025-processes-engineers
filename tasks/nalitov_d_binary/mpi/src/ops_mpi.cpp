#include "nalitov_d_binary/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
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

int64_t Cross(const GridPoint &a, const GridPoint &b, const GridPoint &c) {
  const int64_t abx = static_cast<int64_t>(b.x) - static_cast<int64_t>(a.x);
  const int64_t aby = static_cast<int64_t>(b.y) - static_cast<int64_t>(a.y);
  const int64_t bcx = static_cast<int64_t>(c.x) - static_cast<int64_t>(b.x);
  const int64_t bcy = static_cast<int64_t>(c.y) - static_cast<int64_t>(b.y);
  return (abx * bcy) - (aby * bcx);
}

void TryVisitNeighborExtended(const std::vector<uint8_t> &extended_pixels, int width, int extended_height,
                              int start_row, int ncol, int nrow, std::vector<bool> &visited,
                              std::queue<GridPoint> &frontier) {
  const int ext_nrow = nrow - start_row + 1;
  if (ext_nrow < 0 || ext_nrow >= extended_height) {
    return;
  }
  const size_t neighbor_idx = ToIndex(ncol, ext_nrow, width);
  if (visited[neighbor_idx] || extended_pixels[neighbor_idx] == 0) {
    return;
  }
  visited[neighbor_idx] = true;
  frontier.emplace(ncol, nrow);
}

void BFSCollectGlobal(BinaryImage &image, int start_col, int start_row, std::vector<bool> &visited,
                      std::vector<GridPoint> &out_component) {
  const int width = image.width;
  const int height = image.height;

  const std::array<std::pair<int, int>, 4> k_directions = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  std::queue<GridPoint> frontier;
  frontier.emplace(start_col, start_row);
  visited[ToIndex(start_col, start_row, width)] = true;

  while (!frontier.empty()) {
    const GridPoint current = frontier.front();
    frontier.pop();
    out_component.push_back(current);

    for (const auto &d : k_directions) {
      const int ncol = current.x + d.first;
      const int nrow = current.y + d.second;

      if (ncol < 0 || ncol >= width || nrow < 0 || nrow >= height) {
        continue;
      }

      const size_t nidx = ToIndex(ncol, nrow, width);
      if (visited[nidx] || image.pixels[nidx] == 0) {
        continue;
      }

      visited[nidx] = true;
      frontier.emplace(ncol, nrow);
    }
  }
}

void BFSCollectExtended(const std::vector<uint8_t> &extended_pixels, int width, int extended_height, int start_ext_col,
                        int start_ext_row, int start_row_global, int start_row, int end_row, std::vector<bool> &visited,
                        std::vector<GridPoint> &out_component, bool &out_touches_local, int &out_min_row) {
  const std::array<std::pair<int, int>, 4> k_directions = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  std::queue<GridPoint> frontier;

  frontier.emplace(start_ext_col, start_row_global);
  visited[ToIndex(start_ext_col, start_ext_row, width)] = true;

  out_touches_local = false;
  out_min_row = std::numeric_limits<int>::max();

  const int max_valid_global_row = start_row + (extended_height - 1);

  while (!frontier.empty()) {
    const GridPoint current = frontier.front();
    frontier.pop();
    out_component.push_back(current);

    if (current.y >= start_row && current.y < end_row) {
      out_touches_local = true;
    }
    out_min_row = std::min(out_min_row, current.y);

    for (const auto &d : k_directions) {
      const int ncol = current.x + d.first;
      const int nrow = current.y + d.second;

      if (ncol < 0 || ncol >= width || nrow < 0 || nrow >= max_valid_global_row) {
        continue;
      }

      TryVisitNeighborExtended(extended_pixels, width, extended_height, start_row, ncol, nrow, visited, frontier);
    }
  }
}

void DiscoverGlobalComponents(BinaryImage &image) {
  const int width = image.width;
  const int height = image.height;
  const size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

  std::vector<bool> visited(total_pixels, false);
  image.components.clear();

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const size_t idx = ToIndex(col, row, width);
      if (image.pixels[idx] == 0 || visited[idx]) {
        continue;
      }
      std::vector<GridPoint> component;
      BFSCollectGlobal(image, col, row, visited, component);
      if (!component.empty()) {
        image.components.push_back(std::move(component));
      }
    }
  }
}

std::vector<int> SerializeHullsToPackedPoints(const BinaryImage &output, const std::vector<int> &hull_sizes) {
  const int total_pts = static_cast<int>(std::accumulate(hull_sizes.begin(), hull_sizes.end(), 0));
  std::vector<int> packed_points;
  packed_points.reserve(static_cast<size_t>(total_pts) * 2U);

  for (const auto &hull : output.convex_hulls) {
    for (const auto &pt : hull) {
      packed_points.push_back(pt.x);
      packed_points.push_back(pt.y);
    }
  }
  return packed_points;
}

void ReconstructHullsFromPackedPoints(BinaryImage &output, const std::vector<int> &packed_points,
                                      const std::vector<int> &hull_sizes) {
  output.convex_hulls.clear();
  output.convex_hulls.reserve(hull_sizes.size());

  size_t offset = 0;
  for (int hull_size : hull_sizes) {
    std::vector<GridPoint> hull;
    hull.reserve(static_cast<size_t>(hull_size));
    for (int j = 0; j < hull_size; ++j) {
      const int x = packed_points[offset++];
      const int y = packed_points[offset++];
      hull.emplace_back(x, y);
    }
    output.convex_hulls.push_back(std::move(hull));
  }
}

}  // namespace

NalitovDBinaryMPI::NalitovDBinaryMPI(const InType &in) : full_image_(in), local_image_() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

bool NalitovDBinaryMPI::ValidationImpl() {
  if (GetInput().width <= 0 || GetInput().height <= 0) {
    return false;
  }
  const size_t expected_size = static_cast<size_t>(GetInput().width) * static_cast<size_t>(GetInput().height);
  return GetInput().pixels.size() == expected_size;
}

bool NalitovDBinaryMPI::PreProcessingImpl() {
  if (rank_ == 0) {
    full_image_ = GetInput();
    for (auto &pixel : full_image_.pixels) {
      pixel = pixel > kThreshold ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
    }
  }

  BroadcastDimensions();
  ScatterPixels();
  ThresholdLocalPixels();
  return true;
}

bool NalitovDBinaryMPI::RunImpl() {
  FindLocalComponents();

  local_image_.convex_hulls.clear();
  local_image_.convex_hulls.reserve(local_image_.components.size());

  for (const auto &component : local_image_.components) {
    if (component.empty()) {
      continue;
    }

    if (component.size() <= 2U) {
      local_image_.convex_hulls.push_back(component);
    } else {
      local_image_.convex_hulls.push_back(BuildConvexHull(component));
    }
  }

  CollectGlobalHulls();
  return true;
}

bool NalitovDBinaryMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    GetOutput() = full_image_;
  } else {
    GetOutput() = BinaryImage{};
  }
  BroadcastOutput();
  return true;
}

void NalitovDBinaryMPI::BroadcastDimensions() {
  std::array<int, 2> dims = {0, 0};
  if (rank_ == 0) {
    dims[0] = full_image_.width;
    dims[1] = full_image_.height;
  }

  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_INT, 0, MPI_COMM_WORLD);
  local_image_.width = dims[0];
  local_image_.height = dims[1];
}

void NalitovDBinaryMPI::ScatterPixels() {
  const int width = local_image_.width;
  const int height = local_image_.height;

  counts_.assign(size_, 0);
  displs_.assign(size_, 0);

  const int base_rows = height / size_;
  const int remainder = height % size_;

  int displacement = 0;
  for (int proc = 0; proc < size_; ++proc) {
    const int rows = base_rows + (proc < remainder ? 1 : 0);
    counts_[proc] = rows * width;
    displs_[proc] = displacement;
    displacement += counts_[proc];

    if (proc == rank_) {
      start_row_ = (base_rows * proc) + std::min(proc, remainder);
      end_row_ = start_row_ + rows;
    }
  }

  local_image_.pixels.resize(static_cast<size_t>(counts_[rank_]));

  MPI_Scatterv(rank_ == 0 ? full_image_.pixels.data() : nullptr, counts_.data(), displs_.data(), MPI_BYTE,
               local_image_.pixels.data(), counts_[rank_], MPI_BYTE, 0, MPI_COMM_WORLD);
}

void NalitovDBinaryMPI::ThresholdLocalPixels() {
  local_image_.components.clear();
  local_image_.convex_hulls.clear();
}

void NalitovDBinaryMPI::FindLocalComponents() {
  const int width = local_image_.width;
  const int local_rows = end_row_ - start_row_;

  const int extended_height = local_rows + 2;
  std::vector<uint8_t> extended_pixels(static_cast<size_t>(extended_height) * static_cast<size_t>(width), 0);

  using DiffT = std::vector<uint8_t>::difference_type;

  for (int row = 0; row < local_rows; ++row) {
    const size_t src_offset = static_cast<size_t>(row) * static_cast<size_t>(width);
    const size_t dst_offset = static_cast<size_t>(row + 1) * static_cast<size_t>(width);
    std::copy_n(local_image_.pixels.begin() + static_cast<DiffT>(src_offset), static_cast<size_t>(width),
                extended_pixels.begin() + static_cast<DiffT>(dst_offset));
  }

  ExchangeBoundaryRows(extended_pixels, extended_height);

  std::vector<bool> visited(extended_pixels.size(), false);
  local_image_.components.clear();

  for (int ext_row = 1; ext_row <= local_rows; ++ext_row) {
    for (int col = 0; col < width; ++col) {
      const size_t idx = ToIndex(col, ext_row, width);
      if (extended_pixels[idx] == 0 || visited[idx]) {
        continue;
      }

      const int global_row = start_row_ + ext_row - 1;

      std::vector<GridPoint> component;
      bool touches_local = false;
      int min_row = std::numeric_limits<int>::max();

      BFSCollectExtended(extended_pixels, width, extended_height, col, ext_row, global_row, start_row_, end_row_,
                         visited, component, touches_local, min_row);

      if (!component.empty() && touches_local && min_row >= start_row_) {
        local_image_.components.push_back(std::move(component));
      }
    }
  }
}

void NalitovDBinaryMPI::ExchangeBoundaryRows(std::vector<uint8_t> &extended_pixels, int extended_height) const {
  const int width = local_image_.width;
  const int local_rows = end_row_ - start_row_;

  std::vector<MPI_Request> requests;
  requests.reserve(4);

  std::vector<uint8_t> top_send_buffer;
  std::vector<uint8_t> bottom_send_buffer;

  if (rank_ > 0) {
    requests.emplace_back();
    MPI_Irecv(extended_pixels.data(), width, MPI_BYTE, rank_ - 1, 0, MPI_COMM_WORLD, &requests.back());

    top_send_buffer.resize(static_cast<size_t>(width), 0);
    if (local_rows > 0) {
      std::copy_n(local_image_.pixels.begin(), static_cast<size_t>(width), top_send_buffer.begin());
    }

    requests.emplace_back();
    MPI_Isend(top_send_buffer.data(), width, MPI_BYTE, rank_ - 1, 1, MPI_COMM_WORLD, &requests.back());
  }

  if (rank_ < size_ - 1) {
    requests.emplace_back();
    MPI_Irecv(extended_pixels.data() + (static_cast<size_t>(extended_height - 1) * static_cast<size_t>(width)), width,
              MPI_BYTE, rank_ + 1, 1, MPI_COMM_WORLD, &requests.back());

    bottom_send_buffer.resize(static_cast<size_t>(width), 0);
    if (local_rows > 0) {
      const size_t begin_idx = static_cast<size_t>(local_rows - 1) * static_cast<size_t>(width);
      const auto begin_it = local_image_.pixels.begin() + static_cast<std::vector<uint8_t>::difference_type>(begin_idx);
      std::copy_n(begin_it, static_cast<size_t>(width), bottom_send_buffer.begin());
    }

    requests.emplace_back();
    MPI_Isend(bottom_send_buffer.data(), width, MPI_BYTE, rank_ + 1, 0, MPI_COMM_WORLD, &requests.back());
  }

  if (!requests.empty()) {
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  }
}

void NalitovDBinaryMPI::CollectGlobalHulls() {
  MPI_Gatherv(local_image_.pixels.data(), counts_[rank_], MPI_BYTE, rank_ == 0 ? full_image_.pixels.data() : nullptr,
              counts_.data(), displs_.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    DiscoverGlobalComponents(full_image_);

    full_image_.convex_hulls.clear();
    full_image_.convex_hulls.reserve(full_image_.components.size());

    for (const auto &component : full_image_.components) {
      if (component.empty()) {
        continue;
      }

      if (component.size() <= 2U) {
        full_image_.convex_hulls.push_back(component);
      } else {
        full_image_.convex_hulls.push_back(BuildConvexHull(component));
      }
    }
  }
}

void NalitovDBinaryMPI::BroadcastOutput() {
  BinaryImage &output = GetOutput();

  std::array<int, 2> dims = {0, 0};
  if (rank_ == 0) {
    dims[0] = output.width;
    dims[1] = output.height;
  }
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_INT, 0, MPI_COMM_WORLD);
  if (rank_ != 0) {
    output.width = dims[0];
    output.height = dims[1];
  }

  int hull_count = rank_ == 0 ? static_cast<int>(output.convex_hulls.size()) : 0;
  MPI_Bcast(&hull_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> hull_sizes;
  if (rank_ == 0) {
    hull_sizes.reserve(static_cast<size_t>(hull_count));
    for (const auto &hull : output.convex_hulls) {
      hull_sizes.push_back(static_cast<int>(hull.size()));
    }
  } else {
    hull_sizes.resize(hull_count);
  }

  if (hull_count > 0) {
    MPI_Bcast(hull_sizes.data(), hull_count, MPI_INT, 0, MPI_COMM_WORLD);
  }

  int total_points = 0;
  for (const auto size : hull_sizes) {
    total_points += size;
  }

  std::vector<int> packed_points;
  if (rank_ == 0) {
    packed_points = SerializeHullsToPackedPoints(output, hull_sizes);
  } else {
    packed_points.resize(static_cast<size_t>(total_points) * 2U);
  }

  if (total_points > 0) {
    MPI_Bcast(packed_points.data(), total_points * 2, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (rank_ != 0) {
    ReconstructHullsFromPackedPoints(output, packed_points, hull_sizes);

    output.pixels.assign(static_cast<size_t>(output.width) * static_cast<size_t>(output.height), 0);
    output.components.clear();
  }
}

std::vector<GridPoint> NalitovDBinaryMPI::BuildConvexHull(const std::vector<GridPoint> &points) {
  if (points.size() <= 2U) {
    return points;
  }

  std::vector<GridPoint> sorted_points = points;
  std::ranges::sort(sorted_points, [](const GridPoint &lhs, const GridPoint &rhs) {
    if (lhs.x != rhs.x) {
      return lhs.x < rhs.x;
    }
    return lhs.y < rhs.y;
  });

  const auto unique_range = std::ranges::unique(sorted_points);
  sorted_points.erase(unique_range.begin(), sorted_points.end());

  if (sorted_points.size() <= 2U) {
    return sorted_points;
  }

  std::vector<GridPoint> lower;
  std::vector<GridPoint> upper;
  lower.reserve(sorted_points.size());
  upper.reserve(sorted_points.size());

  for (const auto &pt : sorted_points) {
    while (lower.size() >= 2U && Cross(lower[lower.size() - 2U], lower.back(), pt) <= 0) {
      lower.pop_back();
    }
    lower.push_back(pt);
  }

  for (const auto &sorted_point : std::ranges::reverse_view(sorted_points)) {
    while (upper.size() >= 2U && Cross(upper[upper.size() - 2U], upper.back(), sorted_point) <= 0) {
      upper.pop_back();
    }
    upper.push_back(sorted_point);
  }

  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

}  // namespace nalitov_d_binary
