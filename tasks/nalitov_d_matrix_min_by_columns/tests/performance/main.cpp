#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "nalitov_d_matrix_min_by_columns/common/include/common.hpp"
#include "nalitov_d_matrix_min_by_columns/mpi/include/ops_mpi.hpp"
#include "nalitov_d_matrix_min_by_columns/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nalitov_d_matrix_min_by_columns {

namespace {

inline InType Generate(int64_t i, int64_t j) {
  uint64_t seed = (i * 100000007ULL + j * 1000000009ULL) ^ 42ULL;

  seed ^= seed >> 12;
  seed ^= seed << 25;
  seed ^= seed >> 27;
  uint64_t value = seed * 0x2545F4914F6CDD1DULL;

  auto result = static_cast<InType>((value % 2000001ULL) - 1000000);
  return result;
}

inline std::vector<InType> CalculateExpectedColumnMins(InType n) {
  std::vector<InType> expected_mins(static_cast<size_t>(n), std::numeric_limits<InType>::max());

  for (InType i = 0; i < n; i++) {
    for (InType j = 0; j < n; j++) {
      InType value = Generate(static_cast<int64_t>(i), static_cast<int64_t>(j));
      expected_mins[static_cast<size_t>(j)] = std::min(value, expected_mins[static_cast<size_t>(j)]);
    }
  }

  return expected_mins;
}

}  // anonymous namespace

class NalitovDMinMatrixPerfomanceTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const InType kTestSize_ = 10000;
  InType input_data_{};
  std::vector<InType> expected_mins_;

  void SetUp() override {
    input_data_ = kTestSize_;
    expected_mins_ = CalculateExpectedColumnMins(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      std::cout << "Size mismatch: expected " << input_data_ << ", got " << output_data.size() << '\n';
      return false;
    }

    for (std::size_t j = 0; std::cmp_less(j, output_data.size()); j++) {
      if (output_data[j] != expected_mins_[j]) {
        std::cout << "Value mismatch at column " << j << ": expected " << expected_mins_[j] << ", got "
                  << output_data[j] << '\n';
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NalitovDMinMatrixPerfomanceTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, NalitovDMinMatrixMPI, NalitovDMinMatrixSEQ>(
    PPC_SETTINGS_nalitov_d_matrix_min_by_columns);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NalitovDMinMatrixPerfomanceTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NalitovDMinMatrixPerfomanceTests, kGtestValues, kPerfTestName);

}  // namespace nalitov_d_matrix_min_by_columns
