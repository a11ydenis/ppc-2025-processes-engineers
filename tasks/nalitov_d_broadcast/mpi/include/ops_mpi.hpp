#pragma once

#include <mpi.h>

#include "nalitov_d_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nalitov_d_broadcast {

int NalitovDBroadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

class NalitovDBroadcastMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NalitovDBroadcastMPI(const InType &in);

 private:
  template <typename T>
  bool ProcessVector(const InType &input_data, int proc_rank, int root_proc, MPI_Datatype mpi_dtype);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nalitov_d_broadcast
