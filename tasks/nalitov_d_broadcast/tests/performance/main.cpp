#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <variant>
#include <vector>

#include "nalitov_d_broadcast/common/include/common.hpp"
#include "nalitov_d_broadcast/mpi/include/ops_mpi.hpp"
#include "nalitov_d_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nalitov_d_broadcast {

class NalitovDRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kArraySize_ = 6000000;
  InType test_input_{};

  void SetUp() override {
    std::vector<double> test_data(kArraySize_);
    for (int idx = 0; idx < kArraySize_; ++idx) {
      test_data[idx] = static_cast<double>(idx) * 0.75;
    }
    test_input_ = InType{.data = InTypeVariant{test_data}, .root = 0};
  }

  bool CheckTestOutputData(OutType &result) final {
    if (!std::holds_alternative<std::vector<double>>(test_input_.data)) {
      return false;
    }
    const auto &src_data = std::get<std::vector<double>>(test_input_.data);
    if (!std::holds_alternative<std::vector<double>>(result)) {
      return false;
    }
    const auto &dst_data = std::get<std::vector<double>>(result);
    if (dst_data.size() != src_data.size()) {
      return false;
    }
    const double tolerance = 1e-10;
    for (std::size_t idx = 0; idx < dst_data.size(); ++idx) {
      if (std::fabs(dst_data[idx] - src_data[idx]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return test_input_;
  }
};

TEST_P(NalitovDRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NalitovDBroadcastMPI, NalitovDBroadcastSEQ>(PPC_SETTINGS_nalitov_d_broadcast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NalitovDRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NalitovDRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace nalitov_d_broadcast
