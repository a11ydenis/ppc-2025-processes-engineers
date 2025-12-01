#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "nalitov_d_matrix_min_by_columns/common/include/common.hpp"
#include "nalitov_d_matrix_min_by_columns/mpi/include/ops_mpi.hpp"
#include "nalitov_d_matrix_min_by_columns/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

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

class NalitovDMinMatrixTestProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_mins_ = CalculateExpectedColumnMins(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }

    for (std::size_t j = 0; j < output_data.size(); j++) {
      if (output_data[j] != expected_mins_[j]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
  std::vector<InType> expected_mins_;
};

namespace {

TEST_P(NalitovDMinMatrixTestProcesses, ComputesColumnMinimumsForDiverseSizes) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kFunctionalParams = {
    std::make_tuple(1, "tuple_unit"),  std::make_tuple(2, "tuple_even"),   std::make_tuple(3, "tuple_odd"),
    std::make_tuple(5, "tuple_5"),     std::make_tuple(17, "tuple_prime"), std::make_tuple(64, "tuple_64"),
    std::make_tuple(99, "tuple_99"),   std::make_tuple(100, "tuple_100"),  std::make_tuple(128, "tuple_128"),
    std::make_tuple(256, "tuple_256"), std::make_tuple(512, "tuple_512")};

const auto kTaskMatrix = std::tuple_cat(ppc::util::AddFuncTask<NalitovDMinMatrixMPI, InType>(
                                            kFunctionalParams, PPC_SETTINGS_nalitov_d_matrix_min_by_columns),
                                        ppc::util::AddFuncTask<NalitovDMinMatrixSEQ, InType>(
                                            kFunctionalParams, PPC_SETTINGS_nalitov_d_matrix_min_by_columns));

const auto kParameterizedValues = ppc::util::ExpandToValues(kTaskMatrix);

const auto kFunctionalTestName = NalitovDMinMatrixTestProcesses::PrintFuncTestName<NalitovDMinMatrixTestProcesses>;

INSTANTIATE_TEST_SUITE_P(MinimumColumnSearchSuite, NalitovDMinMatrixTestProcesses, kParameterizedValues,
                         kFunctionalTestName);

template <typename TaskType>
void ExpectFullPipelineSuccess(InType n) {
  auto task = std::make_shared<TaskType>(n);

  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());

  std::vector<InType> expected_mins = CalculateExpectedColumnMins(n);
  OutType output = task->GetOutput();

  ASSERT_EQ(output.size(), static_cast<std::size_t>(n));

  for (std::size_t j = 0; std::cmp_less(j, static_cast<std::size_t>(n)); j++) {
    ASSERT_EQ(output[j], expected_mins[j]) << "Column " << j << ": mismatch";
  }
}

TEST(NalitovDMinMatrixStandalone, SeqPipelineHandlesEdgeSizes) {
  const std::array<InType, 6> k_sizes = {1, 4, 15, 33, 127, 255};
  for (InType size : k_sizes) {
    ExpectFullPipelineSuccess<NalitovDMinMatrixSEQ>(size);
  }
}

TEST(NalitovDMinMatrixStandalone, MpiPipelineHandlesEdgeSizes) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  const std::array<InType, 6> k_sizes = {1, 5, 18, 37, 130, 257};
  for (InType size : k_sizes) {
    ExpectFullPipelineSuccess<NalitovDMinMatrixMPI>(size);
  }
}

TEST(NalitovDMinMatrixValidation, RejectsZeroInputSeq) {
  NalitovDMinMatrixSEQ task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(NalitovDMinMatrixValidation, RejectsZeroInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  NalitovDMinMatrixMPI task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(NalitovDMinMatrixValidation, AcceptsPositiveInputSeq) {
  NalitovDMinMatrixSEQ task(10);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(NalitovDMinMatrixValidation, AcceptsPositiveInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  NalitovDMinMatrixMPI task(10);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

template <typename TaskType>
void RunTaskTwice(TaskType &task, InType n) {
  task.GetInput() = n;
  task.GetOutput().clear();
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  std::vector<InType> expected_mins = CalculateExpectedColumnMins(n);
  OutType output = task.GetOutput();

  ASSERT_EQ(output.size(), static_cast<std::size_t>(n));

  for (std::size_t j = 0; std::cmp_less(j, static_cast<std::size_t>(n)); j++) {
    ASSERT_EQ(output[j], expected_mins[j]) << "Column " << j << " minimum mismatch in reuse test";
  }
}

TEST(NalitovDMinMatrixPipeline, SeqTaskCanBeReusedAcrossRuns) {
  NalitovDMinMatrixSEQ task(4);
  RunTaskTwice(task, 4);
  RunTaskTwice(task, 9);
}

TEST(NalitovDMinMatrixPipeline, MpiTaskCanBeReusedAcrossRuns) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  NalitovDMinMatrixMPI task(6);
  RunTaskTwice(task, 6);
  RunTaskTwice(task, 14);
}

}  // namespace

}  // namespace nalitov_d_matrix_min_by_columns
