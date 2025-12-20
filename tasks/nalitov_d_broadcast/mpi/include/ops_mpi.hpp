#pragma once

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
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nalitov_d_broadcast
