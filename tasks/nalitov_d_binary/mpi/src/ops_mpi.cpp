#include "nalitov_d_binary/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>

#include "nalitov_d_binary/common/include/common.hpp"

namespace nalitov_d_binary {

namespace {

constexpr uint8_t kThreshold = 128;

[[nodiscard]] size_t ToIndex(int x, int y, int width) {
  return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
}

long long Cross(const GridPoint &a, const GridPoint &b, const GridPoint &c) {
  const long long abx = static_cast<long long>(b.x) - static_cast<long long>(a.x);
  const long long aby = static_cast<long long>(b.y) - static_cast<long long>(a.y);
  const long long bcx = static_cast<long long>(c.x) - static_cast<long long>(b.x);
  const long long bcy = static_cast<long long>(c.y) - static_cast<long long>(b.y);
  return (abx * bcy) - (aby * bcx);
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
  int dims[2] = {0, 0};
  if (rank_ == 0) {
    dims[0] = full_image_.width;
    dims[1] = full_image_.height;
  }

  MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
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
      start_row_ = base_rows * proc + std::min(proc, remainder);
      end_row_ = start_row_ + rows;
    }
  }

  local_image_.pixels.resize(static_cast<size_t>(counts_[rank_]));

  MPI_Scatterv(rank_ == 0 ? full_image_.pixels.data() : nullptr, counts_.data(), displs_.data(), MPI_UINT8_T,
               local_image_.pixels.data(), counts_[rank_], MPI_UINT8_T, 0, MPI_COMM_WORLD);
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

  for (int row = 0; row < local_rows; ++row) {
    std::copy_n(local_image_.pixels.begin() + static_cast<long long>(row) * width, static_cast<size_t>(width),
                extended_pixels.begin() + static_cast<long long>(row + 1) * width);
  }

  ExchangeBoundaryRows(extended_pixels, extended_height);

  const int global_rows = local_image_.height;
  std::vector<bool> visited(extended_pixels.size(), false);
  local_image_.components.clear();

  const std::array<std::pair<int, int>, 4> kDirections = {std::make_pair(1, 0), std::make_pair(-1, 0),
                                                          std::make_pair(0, 1), std::make_pair(0, -1)};

  for (int ext_y = 1; ext_y <= local_rows; ++ext_y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = ToIndex(x, ext_y, width);
      if (extended_pixels[idx] == 0 || visited[idx]) {
        continue;
      }

      std::queue<GridPoint> frontier;
      std::vector<GridPoint> component;

      const int global_y = start_row_ + ext_y - 1;
      frontier.emplace(x, global_y);
      visited[idx] = true;

      bool touches_local = false;
      int min_row = std::numeric_limits<int>::max();

      while (!frontier.empty()) {
        const GridPoint current = frontier.front();
        frontier.pop();
        component.push_back(current);

        touches_local |= (current.y >= start_row_ && current.y < end_row_);
        min_row = std::min(min_row, current.y);

        for (const auto &[dx, dy] : kDirections) {
          const int nx = current.x + dx;
          const int ny = current.y + dy;

          if (nx < 0 || nx >= width || ny < 0 || ny >= global_rows) {
            continue;
          }

          const int ext_ny = ny - start_row_ + 1;
          if (ext_ny < 0 || ext_ny >= extended_height) {
            continue;
          }

          const size_t neighbor_idx = ToIndex(nx, ext_ny, width);
          if (visited[neighbor_idx] || extended_pixels[neighbor_idx] == 0) {
            continue;
          }

          visited[neighbor_idx] = true;
          frontier.emplace(nx, ny);
        }
      }

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
    MPI_Irecv(extended_pixels.data(), width, MPI_UINT8_T, rank_ - 1, 0, MPI_COMM_WORLD, &requests.back());

    top_send_buffer.resize(static_cast<size_t>(width), 0);
    if (local_rows > 0) {
      std::copy_n(local_image_.pixels.begin(), static_cast<size_t>(width), top_send_buffer.begin());
    }

    requests.emplace_back();
    MPI_Isend(top_send_buffer.data(), width, MPI_UINT8_T, rank_ - 1, 1, MPI_COMM_WORLD, &requests.back());
  }

  if (rank_ < size_ - 1) {
    requests.emplace_back();
    MPI_Irecv(extended_pixels.data() + static_cast<size_t>(extended_height - 1) * static_cast<size_t>(width), width,
              MPI_UINT8_T, rank_ + 1, 1, MPI_COMM_WORLD, &requests.back());

    bottom_send_buffer.resize(static_cast<size_t>(width), 0);
    if (local_rows > 0) {
      const auto begin_it = local_image_.pixels.begin() + static_cast<long long>(local_rows - 1) * width;
      std::copy_n(begin_it, static_cast<size_t>(width), bottom_send_buffer.begin());
    }

    requests.emplace_back();
    MPI_Isend(bottom_send_buffer.data(), width, MPI_UINT8_T, rank_ + 1, 0, MPI_COMM_WORLD, &requests.back());
  }

  if (!requests.empty()) {
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  }
}

void NalitovDBinaryMPI::CollectGlobalHulls() {
  const int local_hull_count = static_cast<int>(local_image_.convex_hulls.size());
  std::vector<int> hull_counts;
  if (rank_ == 0) {
    hull_counts.resize(size_, 0);
  }

  MPI_Gather(&local_hull_count, 1, MPI_INT, rank_ == 0 ? hull_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    full_image_.convex_hulls = local_image_.convex_hulls;

    for (int proc = 1; proc < size_; ++proc) {
      const int hull_count = hull_counts[proc];
      for (int h = 0; h < hull_count; ++h) {
        int hull_size = 0;
        MPI_Recv(&hull_size, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (hull_size <= 0) {
          continue;
        }
        std::vector<int> buffer(static_cast<size_t>(hull_size) * 2U);
        MPI_Recv(buffer.data(), hull_size * 2, MPI_INT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<GridPoint> hull;
        hull.reserve(static_cast<size_t>(hull_size));
        for (int i = 0; i < hull_size; ++i) {
          hull.emplace_back(buffer[static_cast<size_t>(i) * 2U], buffer[static_cast<size_t>(i) * 2U + 1]);
        }
        full_image_.convex_hulls.push_back(std::move(hull));
      }
    }
  } else {
    for (const auto &hull : local_image_.convex_hulls) {
      const int hull_size = static_cast<int>(hull.size());
      MPI_Send(&hull_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      if (hull_size <= 0) {
        continue;
      }
      std::vector<int> buffer;
      buffer.reserve(static_cast<size_t>(hull_size) * 2U);
      for (const auto &pt : hull) {
        buffer.push_back(pt.x);
        buffer.push_back(pt.y);
      }
      MPI_Send(buffer.data(), hull_size * 2, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
  }
}

void NalitovDBinaryMPI::BroadcastOutput() {
  BinaryImage &output = GetOutput();

  int dims[2] = {0, 0};
  if (rank_ == 0) {
    dims[0] = output.width;
    dims[1] = output.height;
  }
  MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank_ != 0) {
    output.width = dims[0];
    output.height = dims[1];
  }

  int hull_count = rank_ == 0 ? static_cast<int>(output.convex_hulls.size()) : 0;
  MPI_Bcast(&hull_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> hull_sizes;
  if (rank_ == 0) {
    hull_sizes.reserve(hull_count);
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
  if (rank_ == 0) {
    for (const auto size : hull_sizes) {
      total_points += size;
    }
  } else {
    for (const auto size : hull_sizes) {
      total_points += size;
    }
  }

  std::vector<int> packed_points;
  if (rank_ == 0) {
    packed_points.reserve(static_cast<size_t>(total_points) * 2U);
    for (const auto &hull : output.convex_hulls) {
      for (const auto &pt : hull) {
        packed_points.push_back(pt.x);
        packed_points.push_back(pt.y);
      }
    }
  } else {
    packed_points.resize(static_cast<size_t>(total_points) * 2U);
  }

  if (total_points > 0) {
    MPI_Bcast(packed_points.data(), total_points * 2, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (rank_ != 0) {
    output.convex_hulls.clear();
    output.convex_hulls.reserve(hull_count);

    size_t offset = 0;
    for (int i = 0; i < hull_count; ++i) {
      std::vector<GridPoint> hull;
      hull.reserve(static_cast<size_t>(hull_sizes[i]));
      for (int j = 0; j < hull_sizes[i]; ++j) {
        const int x = packed_points[offset++];
        const int y = packed_points[offset++];
        hull.emplace_back(x, y);
      }
      output.convex_hulls.push_back(std::move(hull));
    }
  }

  if (rank_ != 0) {
    output.pixels.assign(static_cast<size_t>(output.width) * static_cast<size_t>(output.height), 0);
    output.components.clear();
  }
}

std::vector<GridPoint> NalitovDBinaryMPI::BuildConvexHull(const std::vector<GridPoint> &points) {
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

  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (upper.size() >= 2U && Cross(upper[upper.size() - 2U], upper.back(), *it) <= 0) {
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
