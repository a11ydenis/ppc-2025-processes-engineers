#include "nalitov_d_matrix_min_by_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "nalitov_d_matrix_min_by_columns/common/include/common.hpp"

namespace nalitov_d_matrix_min_by_columns {

NalitovDMinMatrixMPI::NalitovDMinMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool NalitovDMinMatrixMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput().empty());
}

bool NalitovDMinMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput());
  return true;
}

bool NalitovDMinMatrixMPI::RunImpl() {
  auto generate = [](int64_t i, int64_t j) -> InType {
    uint64_t seed = (i * 100000007ULL + j * 1000000009ULL) ^ 42ULL;

    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    uint64_t value = seed * 0x2545F4914F6CDD1DULL;

    return static_cast<InType>((value % 2000001) - 1000000);
  };

  InType n = GetInput();
  if (n == 0) {
    return false;
  }

  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_process = n / size;
  int leftover = n % size;
  int process_rows_number = rows_per_process + (rank < leftover ? 1 : 0);

  int first_row = (rank * rows_per_process) + std::min(rank, leftover);
  int last_row = first_row + process_rows_number;

  std::vector<InType> local_min_columns(static_cast<size_t>(n), std::numeric_limits<InType>::max());

  for (InType i = first_row; i < last_row; i++) {
    for (InType j = 0; j < n; j++) {
      InType val = generate(static_cast<int64_t>(i), static_cast<int64_t>(j));
      local_min_columns[static_cast<size_t>(j)] = std::min(local_min_columns[static_cast<size_t>(j)], val);
    }
  }

  std::vector<InType> global_min_columns(static_cast<size_t>(n), std::numeric_limits<InType>::max());

  MPI_Reduce(local_min_columns.data(), global_min_columns.data(), static_cast<int>(n), MPI_INT, MPI_MIN, 0,
             MPI_COMM_WORLD);

  MPI_Bcast(global_min_columns.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = std::move(global_min_columns);

  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n));
}

bool NalitovDMinMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(GetInput()));
}

}  // namespace nalitov_d_matrix_min_by_columns
