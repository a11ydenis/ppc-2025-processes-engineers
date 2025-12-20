#include "nalitov_d_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <variant>
#include <vector>

#include "nalitov_d_broadcast/common/include/common.hpp"

namespace nalitov_d_broadcast {

namespace {

void DistributeData(void *data_ptr, int elem_count, MPI_Datatype mpi_dtype, int root_proc, MPI_Comm comm) {
  if (elem_count == 0) {
    MPI_Barrier(comm);
    return;
  }

  int comm_size = 0;
  MPI_Comm_size(comm, &comm_size);

  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);

  if (comm_size <= 1) {
    return;
  }

  int dtype_size = 0;
  MPI_Type_size(mpi_dtype, &dtype_size);
  size_t buffer_size = static_cast<size_t>(elem_count) * static_cast<size_t>(dtype_size);

  std::vector<unsigned char> work_buffer(buffer_size);

  if (my_rank == root_proc) {
    std::memcpy(work_buffer.data(), data_ptr, buffer_size);
  }

  int tree_levels = 0;
  int remaining = comm_size;
  while (remaining > 1) {
    remaining >>= 1;
    tree_levels++;
  }

  for (int current_level = 0; current_level < tree_levels; current_level++) {
    int level_step = 1 << current_level;
    int mapped_rank = (my_rank - root_proc + comm_size) % comm_size;

    if ((mapped_rank % (2 * level_step)) == 0) {
      int dest_mapped = mapped_rank + level_step;
      if (dest_mapped < comm_size) {
        int dest_rank = (dest_mapped + root_proc) % comm_size;
        MPI_Send(work_buffer.data(), elem_count, mpi_dtype, dest_rank, 0, comm);
      }
    } else if ((mapped_rank % level_step) == 0) {
      int src_mapped = mapped_rank - level_step;
      int src_rank = (src_mapped + root_proc) % comm_size;
      MPI_Status status{};
      MPI_Recv(work_buffer.data(), elem_count, mpi_dtype, src_rank, 0, comm, &status);
    }
  }

  if (my_rank != root_proc) {
    std::memcpy(data_ptr, work_buffer.data(), buffer_size);
  }
}

void DistributeInteger(int *val, int root_proc, MPI_Comm comm) {
  DistributeData(static_cast<void *>(val), 1, MPI_INT, root_proc, comm);
}

}  // namespace

NalitovDBroadcastMPI::NalitovDBroadcastMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  int proc_rank = 0;
  int init_flag = 0;
  MPI_Initialized(&init_flag);

  if (init_flag != 0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  }

  if (proc_rank == 0) {
    if (std::holds_alternative<std::vector<int>>(in.data)) {
      const auto &src_vec = std::get<std::vector<int>>(in.data);
      GetOutput() = InTypeVariant{std::vector<int>(src_vec.size(), 0)};
    } else if (std::holds_alternative<std::vector<float>>(in.data)) {
      const auto &src_vec = std::get<std::vector<float>>(in.data);
      GetOutput() = InTypeVariant{std::vector<float>(src_vec.size(), 0.0F)};
    } else if (std::holds_alternative<std::vector<double>>(in.data)) {
      const auto &src_vec = std::get<std::vector<double>>(in.data);
      GetOutput() = InTypeVariant{std::vector<double>(src_vec.size(), 0.0)};
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }
}

bool NalitovDBroadcastMPI::ValidationImpl() {
  int proc_rank = 0;
  int init_flag = 0;
  MPI_Initialized(&init_flag);

  if (init_flag != 0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    int comm_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (proc_rank == 0) {
      const auto &input_data = GetInput();
      bool valid_type = false;

      if (std::holds_alternative<std::vector<int>>(input_data.data)) {
        valid_type = true;
      } else if (std::holds_alternative<std::vector<float>>(input_data.data)) {
        valid_type = true;
      } else if (std::holds_alternative<std::vector<double>>(input_data.data)) {
        valid_type = true;
      }

      if (!valid_type) {
        return false;
      }

      if (input_data.root < 0 || input_data.root >= comm_size) {
        return false;
      }
    }
  }
  return true;
}

bool NalitovDBroadcastMPI::PreProcessingImpl() {
  return true;
}

bool NalitovDBroadcastMPI::RunImpl() {
  try {
    const auto &input_data = GetInput();
    int proc_rank = 0;
    int root_proc = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    if (proc_rank == 0) {
      root_proc = input_data.root;
    }
    DistributeInteger(&root_proc, 0, MPI_COMM_WORLD);

    if (std::holds_alternative<std::vector<int>>(input_data.data)) {
      return ProcessVector<int>(input_data, proc_rank, root_proc, MPI_INT);
    }

    if (std::holds_alternative<std::vector<float>>(input_data.data)) {
      return ProcessVector<float>(input_data, proc_rank, root_proc, MPI_FLOAT);
    }

    if (std::holds_alternative<std::vector<double>>(input_data.data)) {
      return ProcessVector<double>(input_data, proc_rank, root_proc, MPI_DOUBLE);
    }

    return false;
  } catch (...) {
    return false;
  }
}

template <typename T>
bool NalitovDBroadcastMPI::ProcessVector(const InType &input_data, int proc_rank, int root_proc,
                                         MPI_Datatype mpi_dtype) {
  int elem_count = 0;
  if (proc_rank == 0) {
    if (std::holds_alternative<std::vector<T>>(input_data.data)) {
      elem_count = static_cast<int>(std::get<std::vector<T>>(input_data.data).size());
    } else {
      return false;
    }
  }

  DistributeInteger(&elem_count, 0, MPI_COMM_WORLD);

  if (elem_count == 0) {
    return true;
  }

  auto &output_result = GetOutput();
  auto &dest_buffer = std::get<std::vector<T>>(output_result);

  if (static_cast<int>(dest_buffer.size()) != elem_count) {
    dest_buffer.resize(elem_count);
  }

  if (proc_rank == 0) {
    const auto &src_buffer = std::get<std::vector<T>>(input_data.data);
    std::ranges::copy(src_buffer, dest_buffer.begin());
  }

  DistributeData(dest_buffer.data(), elem_count, mpi_dtype, root_proc, MPI_COMM_WORLD);

  return true;
}

bool NalitovDBroadcastMPI::PostProcessingImpl() {
  return true;
}

template bool NalitovDBroadcastMPI::ProcessVector<int>(const InType &input, int rank, int root, MPI_Datatype mpi_type);
template bool NalitovDBroadcastMPI::ProcessVector<float>(const InType &input, int rank, int root,
                                                         MPI_Datatype mpi_type);
template bool NalitovDBroadcastMPI::ProcessVector<double>(const InType &input, int rank, int root,
                                                          MPI_Datatype mpi_type);

}  // namespace nalitov_d_broadcast
