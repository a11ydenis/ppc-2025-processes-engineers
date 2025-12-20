#pragma once

#include <mpi.h>

#include "nalitov_d_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nalitov_d_broadcast {

class NalitovDBroadcastMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NalitovDBroadcastMPI(const InType &in);

 private:
  template <typename T>
  bool ProcessVector(const InType &input, int rank, int root, MPI_Datatype mpi_type);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nalitov_d_broadcast
