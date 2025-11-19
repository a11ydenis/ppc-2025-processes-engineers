#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace nalitov_d_matrix_min_by_columns {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nalitov_d_matrix_min_by_columns
