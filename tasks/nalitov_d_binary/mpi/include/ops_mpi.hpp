#pragma once

#include <cstdint>
#include <vector>

#include "nalitov_d_binary/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nalitov_d_binary {

class NalitovDBinaryMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit NalitovDBinaryMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastDimensions();
  void ScatterPixels();
  void ThresholdLocalPixels();
  void FindLocalComponents();
  void ExchangeBoundaryRows(std::vector<uint8_t> &extended_pixels, int extended_height) const;
  void CollectGlobalHulls();
  void BroadcastOutput();
  static std::vector<GridPoint> BuildConvexHull(const std::vector<GridPoint> &points);

  BinaryImage full_image_;
  BinaryImage local_image_;
  int rank_{0};
  int size_{1};
  int start_row_{0};
  int end_row_{0};
  std::vector<int> counts_;
  std::vector<int> displs_;
};

}  // namespace nalitov_d_binary
