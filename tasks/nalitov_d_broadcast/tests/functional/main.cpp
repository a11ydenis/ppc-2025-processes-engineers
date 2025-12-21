#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "nalitov_d_broadcast/common/include/common.hpp"
#include "nalitov_d_broadcast/mpi/include/ops_mpi.hpp"
#include "nalitov_d_broadcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace nalitov_d_broadcast {

class NalitovDRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  NalitovDRunFuncTestsProcesses() : mpi_dtype_(MPI_DATATYPE_NULL) {}

  void SetUp() override {
    auto param = GetParam();
    TestType test_param = std::get<2>(param);
    int test_id = std::get<0>(test_param);
    std::string task_name = std::get<1>(param);
    uses_mpi_ = (task_name.find("mpi") != std::string::npos);

    int world_size = 1;
    if (uses_mpi_) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      if (world_size <= 0) {
        world_size = 1;
      }
    }

    const int root = (uses_mpi_ && world_size > 0) ? (test_id % world_size) : 0;

    switch (test_id) {
      case 1: {
        std::vector<double> input = {2.5, 3.7, 4.9, 6.1, 7.3};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_double_ = input;
        mpi_dtype_ = MPI_DOUBLE;
      } break;
      case 2: {
        std::vector<int> input = {10, 20, 30, 40, 50};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_int_ = input;
        mpi_dtype_ = MPI_INT;
      } break;
      case 3: {
        std::vector<float> input = {0.5F, 1.5F, 2.5F, 3.5F, 4.5F};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_float_ = input;
        mpi_dtype_ = MPI_FLOAT;
      } break;
      case 4: {
        std::vector<int> input = {5, -10, 15, -20, 25};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_int_ = input;
        mpi_dtype_ = MPI_INT;
      } break;
      case 5: {
        std::vector<double> input = {-2.1, -3.2, -4.3, -5.4, -6.5};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_double_ = input;
        mpi_dtype_ = MPI_DOUBLE;
      } break;
      case 6: {
        std::vector<float> input = {-0.7F, 1.3F, -2.1F, 3.9F, -4.5F};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_float_ = input;
        mpi_dtype_ = MPI_FLOAT;
      } break;
      case 7: {
        std::vector<int> input = {-3, 6, -9, 12, -15};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_int_ = input;
        mpi_dtype_ = MPI_INT;
      } break;
      case 8: {
        std::vector<double> input = {0.1, -0.2, 0.3, -0.4, 0.5};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_double_ = input;
        mpi_dtype_ = MPI_DOUBLE;
      } break;
      case 9: {
        std::vector<float> input = {2.2F, -3.3F, 4.4F, -5.5F, 6.6F};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_float_ = input;
        mpi_dtype_ = MPI_FLOAT;
      } break;
      case 10: {
        std::vector<double> input = {};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_double_ = input;
        mpi_dtype_ = MPI_DOUBLE;
      } break;
      case 11: {
        std::vector<int> input = {0, 0, 0, 0};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_int_ = input;
        mpi_dtype_ = MPI_INT;
      } break;
      case 12: {
        std::vector<double> input = {1.11, 2.22, 3.33, 4.44, 5.55};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_double_ = input;
        mpi_dtype_ = MPI_DOUBLE;
      } break;
      case 13: {
        std::vector<float> input = {1.25F, 2.75F, 3.25F, 4.75F, 5.25F};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_float_ = input;
        mpi_dtype_ = MPI_FLOAT;
      } break;
      case 14: {
        std::vector<int> input = {7, 14, 21, 28, 35};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_int_ = input;
        mpi_dtype_ = MPI_INT;
      } break;
      case 15: {
        std::vector<double> input = {-1.5, 2.5, -3.5, 4.5, -5.5};
        input_data_ = InType{.data = InTypeVariant{input}, .root = root};
        reference_double_ = input;
        mpi_dtype_ = MPI_DOUBLE;
      } break;
      case 16: {
        const int arr_size = 1000;
        std::vector<double> input(arr_size);
        std::vector<int> input_int(arr_size);
        std::vector<float> input_float(arr_size);
        if (uses_mpi_) {
          if (task_name.find("int") != std::string::npos) {
            for (int i = 0; i < arr_size; i++) {
              input_int[i] = i * 3;
            }
            input_data_ = InType{.data = InTypeVariant{input_int}, .root = root};
            reference_int_ = input_int;
            mpi_dtype_ = MPI_INT;
          } else if (task_name.find("float") != std::string::npos) {
            for (int i = 0; i < arr_size; i++) {
              input_float[i] = static_cast<float>(i) * 2.3F;
            }
            input_data_ = InType{.data = InTypeVariant{input_float}, .root = root};
            reference_float_ = input_float;
            mpi_dtype_ = MPI_FLOAT;
          } else {
            for (int i = 0; i < arr_size; i++) {
              input[i] = static_cast<double>(i) * 2.1;
            }
            input_data_ = InType{.data = InTypeVariant{input}, .root = root};
            reference_double_ = input;
            mpi_dtype_ = MPI_DOUBLE;
          }
        } else {
          for (int i = 0; i < arr_size; i++) {
            input[i] = static_cast<double>(i) * 2.1;
          }
          input_data_ = InType{.data = InTypeVariant{input}, .root = 0};
          reference_double_ = input;
          mpi_dtype_ = MPI_DOUBLE;
        }
      } break;
      default: {
        std::vector<double> input = {2.5, 3.7, 4.9, 6.1, 7.3};
        input_data_ = InType{.data = InTypeVariant{input}, .root = 0};
        reference_double_ = input;
        mpi_dtype_ = MPI_DOUBLE;
      } break;
    }
  }

 private:
  bool ValidateSequentialResult(const OutType &result) {
    if (std::holds_alternative<std::vector<int>>(input_data_.data)) {
      const auto &src = std::get<std::vector<int>>(input_data_.data);
      if (!std::holds_alternative<std::vector<int>>(result)) {
        return false;
      }
      const auto &dst = std::get<std::vector<int>>(result);
      if (dst.size() != src.size()) {
        return false;
      }
      for (std::size_t idx = 0; idx < dst.size(); ++idx) {
        if (dst[idx] != src[idx]) {
          return false;
        }
      }
      return true;
    }
    if (std::holds_alternative<std::vector<float>>(input_data_.data)) {
      const auto &src = std::get<std::vector<float>>(input_data_.data);
      if (!std::holds_alternative<std::vector<float>>(result)) {
        return false;
      }
      const auto &dst = std::get<std::vector<float>>(result);
      if (dst.size() != src.size()) {
        return false;
      }
      const float tolerance = 1e-5F;
      for (std::size_t idx = 0; idx < dst.size(); ++idx) {
        if (std::fabs(dst[idx] - src[idx]) > tolerance) {
          return false;
        }
      }
      return true;
    }
    if (std::holds_alternative<std::vector<double>>(input_data_.data)) {
      const auto &src = std::get<std::vector<double>>(input_data_.data);
      if (!std::holds_alternative<std::vector<double>>(result)) {
        return false;
      }
      const auto &dst = std::get<std::vector<double>>(result);
      if (dst.size() != src.size()) {
        return false;
      }
      const double tolerance = 1e-10;
      for (std::size_t idx = 0; idx < dst.size(); ++idx) {
        if (std::fabs(dst[idx] - src[idx]) > tolerance) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  bool ValidateParallelResult(const OutType &result) {
    int init_status = 0;
    MPI_Initialized(&init_status);
    if (init_status == 0) {
      return true;
    }

    if (std::holds_alternative<std::vector<int>>(input_data_.data)) {
      const auto &expected = reference_int_;
      if (!std::holds_alternative<std::vector<int>>(result)) {
        return false;
      }
      const auto &actual = std::get<std::vector<int>>(result);
      if (actual.size() != expected.size()) {
        return false;
      }
      for (std::size_t idx = 0; idx < actual.size(); ++idx) {
        if (actual[idx] != expected[idx]) {
          return false;
        }
      }
      return true;
    }
    if (std::holds_alternative<std::vector<float>>(input_data_.data)) {
      const auto &expected = reference_float_;
      if (!std::holds_alternative<std::vector<float>>(result)) {
        return false;
      }
      const auto &actual = std::get<std::vector<float>>(result);
      if (actual.size() != expected.size()) {
        return false;
      }
      const float tolerance = 1e-5F;
      for (std::size_t idx = 0; idx < actual.size(); ++idx) {
        if (std::fabs(actual[idx] - expected[idx]) > tolerance) {
          return false;
        }
      }
      return true;
    }
    if (std::holds_alternative<std::vector<double>>(input_data_.data)) {
      const auto &expected = reference_double_;
      if (!std::holds_alternative<std::vector<double>>(result)) {
        return false;
      }
      const auto &actual = std::get<std::vector<double>>(result);
      if (actual.size() != expected.size()) {
        return false;
      }
      const double tolerance = 1e-10;
      for (std::size_t idx = 0; idx < actual.size(); ++idx) {
        if (std::fabs(actual[idx] - expected[idx]) > tolerance) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

 public:
  bool CheckTestOutputData(OutType &output_data) final {
    try {
      if (!uses_mpi_) {
        return ValidateSequentialResult(output_data);
      }
      return ValidateParallelResult(output_data);
    } catch (...) {
      return false;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<double> reference_double_;
  std::vector<int> reference_int_;
  std::vector<float> reference_float_;
  MPI_Datatype mpi_dtype_;
  bool uses_mpi_ = false;
};

namespace {

TEST_P(NalitovDRunFuncTestsProcesses, BroadcastTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
    std::make_tuple(1, "doubles_positive"),   std::make_tuple(2, "ints_positive"),
    std::make_tuple(3, "floats_positive"),    std::make_tuple(4, "ints_mixed"),
    std::make_tuple(5, "doubles_negative"),   std::make_tuple(6, "floats_mixed"),
    std::make_tuple(7, "ints_alternating"),   std::make_tuple(8, "doubles_small"),
    std::make_tuple(9, "floats_alternating"), std::make_tuple(10, "empty_array"),
    std::make_tuple(11, "ints_zero"),         std::make_tuple(12, "doubles_precise"),
    std::make_tuple(13, "floats_precise"),    std::make_tuple(14, "ints_large"),
    std::make_tuple(15, "doubles_mixed"),     std::make_tuple(16, "large_dataset"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<NalitovDBroadcastSEQ, InType>(kTestParam, PPC_SETTINGS_nalitov_d_broadcast),
                   ppc::util::AddFuncTask<NalitovDBroadcastMPI, InType>(kTestParam, PPC_SETTINGS_nalitov_d_broadcast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NalitovDRunFuncTestsProcesses::PrintFuncTestName<NalitovDRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(BroadcastTests, NalitovDRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nalitov_d_broadcast
