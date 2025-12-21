#pragma once

#include <vector>

#include "nalitov_d_binary/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nalitov_d_binary {

class NalitovDBinarySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NalitovDBinarySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ThresholdImage();
  void DiscoverComponents();
  static std::vector<GridPoint> BuildConvexHull(const std::vector<GridPoint> &points);

  BinaryImage working_image_;
};

}  // namespace nalitov_d_binary
