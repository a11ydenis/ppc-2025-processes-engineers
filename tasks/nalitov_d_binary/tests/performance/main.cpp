#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nalitov_d_binary/common/include/common.hpp"
#include "nalitov_d_binary/mpi/include/ops_mpi.hpp"
#include "nalitov_d_binary/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nalitov_d_binary {

namespace {

BinaryImage MakePerfImage(int size) {
  BinaryImage image;
  image.width = size;
  image.height = size;
  image.pixels.assign(static_cast<size_t>(size) * static_cast<size_t>(size), 0);

  const int center_x = size / 2;
  const int center_y = size / 2;
  const int radius = size / 5;

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if ((dx * dx) + (dy * dy) <= radius * radius) {
        const int px = center_x + dx;
        const int py = center_y + dy;
        if (px >= 0 && px < size && py >= 0 && py < size) {
          image.pixels[(static_cast<size_t>(py) * static_cast<size_t>(size)) + static_cast<size_t>(px)] = 255;
        }
      }
    }
  }

  for (int idx = 0; idx < size; ++idx) {
    image.pixels[(static_cast<size_t>(idx) * static_cast<size_t>(size)) + static_cast<size_t>(idx)] = 255;
    image.pixels[(static_cast<size_t>(idx) * static_cast<size_t>(size)) + static_cast<size_t>(size - 1 - idx)] = 255;
  }

  for (int row = size / 4; row < (3 * size) / 4; row += 3) {
    for (int col = 0; col < size; ++col) {
      image.pixels[(static_cast<size_t>(row) * static_cast<size_t>(size)) + static_cast<size_t>(col)] = 255;
    }
  }

  return image;
}

bool ValidateHull(const std::vector<GridPoint> &hull, int width, int height) {
  if (hull.empty()) {
    return false;
  }

  for (const auto &point : hull) {
    if (point.x < 0 || point.x >= width || point.y < 0 || point.y >= height) {
      return false;
    }
  }

  if (hull.size() >= 3U) {
    int64_t orientation = 0;
    const size_t count = hull.size();

    for (size_t idx = 0; idx < count; ++idx) {
      const GridPoint &a = hull[idx];
      const GridPoint &b = hull[(idx + 1) % count];
      const GridPoint &c = hull[(idx + 2) % count];

      const int64_t cross = ((static_cast<int64_t>(b.x) - static_cast<int64_t>(a.x)) *
                             (static_cast<int64_t>(c.y) - static_cast<int64_t>(b.y))) -
                            ((static_cast<int64_t>(b.y) - static_cast<int64_t>(a.y)) *
                             (static_cast<int64_t>(c.x) - static_cast<int64_t>(b.x)));

      if (cross != 0) {
        if (orientation == 0) {
          orientation = cross;
        } else if ((orientation > 0) != (cross > 0)) {
          return false;
        }
      }
    }
  }

  if (hull.size() == 2U && hull[0] == hull[1]) {
    return false;
  }

  return true;
}

bool ValidateOutput(const BinaryImage &output_data) {
  return std::ranges::all_of(output_data.convex_hulls, [&](const std::vector<GridPoint> &hull) {
    return ValidateHull(hull, output_data.width, output_data.height);
  });
}

}  // namespace

class NalitovDBinaryPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.width == input_data_.width && output_data.height == input_data_.height &&
           ValidateOutput(output_data);
  }

  InType GetTestInputData() final {
    input_data_ = MakePerfImage(384);
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(NalitovDBinaryPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NalitovDBinaryMPI, NalitovDBinarySEQ>(PPC_SETTINGS_nalitov_d_binary);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NalitovDBinaryPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NalitovDBinaryPerfTests, kGtestValues, kPerfTestName);

}  // namespace nalitov_d_binary
