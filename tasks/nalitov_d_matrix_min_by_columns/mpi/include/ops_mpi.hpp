#pragma once

#include "nalitov_d_matrix_min_by_columns/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nalitov_d_matrix_min_by_columns {

class NalitovDMinMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NalitovDMinMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nalitov_d_matrix_min_by_columns
