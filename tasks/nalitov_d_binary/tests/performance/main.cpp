#include <gtest/gtest.h>

#include "nalitov_d_binary/common/include/common.hpp"
#include "nalitov_d_binary/mpi/include/ops_mpi.hpp"
#include "nalitov_d_binary/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nalitov_d_binary {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NalitovDBinaryMPI, NalitovDBinarySEQ>(PPC_SETTINGS_nalitov_d_binary);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace nalitov_d_binary
