#pragma once

#include "nalitov_d_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nalitov_d_broadcast {

class NalitovDBroadcastSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NalitovDBroadcastSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nalitov_d_broadcast
