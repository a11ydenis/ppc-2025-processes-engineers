#include "nalitov_d_broadcast/seq/include/ops_seq.hpp"

#include <cstring>
#include <stdexcept>
#include <variant>
#include <vector>

#include "nalitov_d_broadcast/common/include/common.hpp"

namespace nalitov_d_broadcast {

namespace {

template <typename T>
bool TransferData(const InType &src, OutType &dst) {
  const auto &src_data = std::get<std::vector<T>>(src.data);
  auto &dst_data = std::get<std::vector<T>>(dst);

  if (dst_data.size() != src_data.size()) {
    dst_data.resize(src_data.size());
  }

  if (!src_data.empty()) {
    std::memcpy(dst_data.data(), src_data.data(), src_data.size() * sizeof(T));
  }
  return true;
}

}  // namespace

NalitovDBroadcastSEQ::NalitovDBroadcastSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  if (std::holds_alternative<std::vector<int>>(in.data)) {
    const auto &src = std::get<std::vector<int>>(in.data);
    GetOutput() = InTypeVariant{std::vector<int>(src)};
  } else if (std::holds_alternative<std::vector<float>>(in.data)) {
    const auto &src = std::get<std::vector<float>>(in.data);
    GetOutput() = InTypeVariant{std::vector<float>(src)};
  } else if (std::holds_alternative<std::vector<double>>(in.data)) {
    const auto &src = std::get<std::vector<double>>(in.data);
    GetOutput() = InTypeVariant{std::vector<double>(src)};
  } else {
    throw std::runtime_error("Unsupported data type");
  }
}

bool NalitovDBroadcastSEQ::ValidationImpl() {
  const auto &src_data = GetInput();

  if (std::holds_alternative<std::vector<int>>(src_data.data)) {
    return true;
  }
  if (std::holds_alternative<std::vector<float>>(src_data.data)) {
    return true;
  }
  if (std::holds_alternative<std::vector<double>>(src_data.data)) {
    return true;
  }

  return false;
}

bool NalitovDBroadcastSEQ::PreProcessingImpl() {
  return true;
}

bool NalitovDBroadcastSEQ::RunImpl() {
  try {
    const auto &src_data = GetInput();
    auto &dst_data = GetOutput();

    if (std::holds_alternative<std::vector<int>>(src_data.data)) {
      return TransferData<int>(src_data, dst_data);
    }
    if (std::holds_alternative<std::vector<float>>(src_data.data)) {
      return TransferData<float>(src_data, dst_data);
    }
    if (std::holds_alternative<std::vector<double>>(src_data.data)) {
      return TransferData<double>(src_data, dst_data);
    }

    return false;
  } catch (...) {
    return false;
  }
}

bool NalitovDBroadcastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nalitov_d_broadcast
