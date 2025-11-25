#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "nalitov_d_matrix_min_by_columns/common/include/common.hpp"
#include "nalitov_d_matrix_min_by_columns/mpi/include/ops_mpi.hpp"
#include "nalitov_d_matrix_min_by_columns/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nalitov_d_matrix_min_by_columns {

inline auto generate_value = [](int64_t i, int64_t j) -> InType {
  uint64_t seed = (i * 100000007ULL + j * 1000000009ULL) ^ 42ULL;

  seed ^= seed >> 12;
  seed ^= seed << 25;
  seed ^= seed >> 27;
  uint64_t val = seed * 0x2545F4914F6CDD1DULL;

  return static_cast<InType>((val % 2000001) - 1000000);
};

inline std::vector<InType> CalculateExpectedColumnMins(InType n) {
  std::vector<InType> expected_mins(static_cast<size_t>(n), std::numeric_limits<InType>::max());

  for (InType i = 0; i < n; i++) {
    for (InType j = 0; j < n; j++) {
      InType val = generate_value(static_cast<int64_t>(i), static_cast<int64_t>(j));
      if (val < expected_mins[static_cast<size_t>(j)]) {
        expected_mins[static_cast<size_t>(j)] = val;
      }
    }
  }

  return expected_mins;
}

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
      std::cout << "Size mismatch: expected " << input_data_ << ", got " << output_data.size() << std::endl;
      return false;
    }

    for (std::size_t j = 0; j < output_data.size(); j++) {
      if (output_data[j] != expected_mins_[j]) {
        std::cout << "Value mismatch at column " << j << ": expected " << expected_mins_[j] << ", got "
                  << output_data[j] << std::endl;
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
