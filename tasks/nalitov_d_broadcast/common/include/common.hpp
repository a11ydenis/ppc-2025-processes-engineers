#pragma once

#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "task/include/task.hpp"

namespace nalitov_d_broadcast {

using InTypeVariant = std::variant<std::vector<int>, std::vector<float>, std::vector<double>>;

struct InType {
  InTypeVariant data;
  int root{0};
};

using OutType = InTypeVariant;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nalitov_d_broadcast
