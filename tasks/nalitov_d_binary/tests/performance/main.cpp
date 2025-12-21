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

  const int cx = size / 2;
  const int cy = size / 2;
  const int radius = size / 5;

  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      if ((x * x) + (y * y) <= radius * radius) {
        const int px = cx + x;
        const int py = cy + y;
        if (px >= 0 && px < size && py >= 0 && py < size) {
          image.pixels[static_cast<size_t>(py) * static_cast<size_t>(size) + static_cast<size_t>(px)] = 255;
        }
      }
    }
  }

  for (int i = 0; i < size; ++i) {
    image.pixels[static_cast<size_t>(i) * static_cast<size_t>(size) + static_cast<size_t>(i)] = 255;
    image.pixels[static_cast<size_t>(i) * static_cast<size_t>(size) + static_cast<size_t>(size - 1 - i)] = 255;
  }

  for (int y = size / 4; y < 3 * size / 4; y += 3) {
    for (int x = 0; x < size; ++x) {
      image.pixels[static_cast<size_t>(y) * static_cast<size_t>(size) + static_cast<size_t>(x)] = 255;
    }
  }

  return image;
}

bool ValidateHull(const std::vector<GridPoint> &hull, int width, int height) {
  if (hull.empty()) {
    return false;
  }

  for (const auto &pt : hull) {
    if (pt.x < 0 || pt.x >= width || pt.y < 0 || pt.y >= height) {
      return false;
    }
  }

  if (hull.size() >= 3U) {
    long long orientation = 0;
    const size_t n = hull.size();
    for (size_t i = 0; i < n; ++i) {
      const GridPoint &a = hull[i];
      const GridPoint &b = hull[(i + 1) % n];
      const GridPoint &c = hull[(i + 2) % n];
      const long long cross = (static_cast<long long>(b.x) - a.x) * (static_cast<long long>(c.y) - b.y) -
                              (static_cast<long long>(b.y) - a.y) * (static_cast<long long>(c.x) - b.x);
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
  for (const auto &hull : output_data.convex_hulls) {
    if (!ValidateHull(hull, output_data.width, output_data.height)) {
      return false;
    }
  }
  return true;
}

}  // namespace

class NalitovDBinaryPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {}

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.width == input_data_.width && output_data.height == input_data_.height &&
           ValidateOutput(output_data);
  }

  InType GetTestInputData() final {
    input_data_ = MakePerfImage(256);
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
