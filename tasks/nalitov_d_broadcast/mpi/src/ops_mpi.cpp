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

constexpr int kBroadcastTag = 0;

int ValidateBroadcastArgs(const void *buffer, int count, int root, int comm_size) {
  if (count < 0) {
    return MPI_ERR_COUNT;
  }

  if (comm_size <= 0) {
    return MPI_ERR_COMM;
  }

  if (root < 0 || root >= comm_size) {
    return MPI_ERR_ROOT;
  }

  if (count == 0) {
    return MPI_SUCCESS;
  }

  if (buffer == nullptr) {
    return MPI_ERR_BUFFER;
  }

  return MPI_SUCCESS;
}

int TreeBroadcastStep(void *buffer, int count, MPI_Datatype datatype, int root, int virtual_rank, int mask,
                      int comm_size, MPI_Comm comm) {
  if (virtual_rank < mask) {
    const int dest_virtual = virtual_rank + mask;
    if (dest_virtual >= comm_size) {
      return MPI_SUCCESS;
    }

    const int dest_rank = (dest_virtual + root) % comm_size;
    return MPI_Send(buffer, count, datatype, dest_rank, kBroadcastTag, comm);
  }

  if (virtual_rank < (mask << 1)) {
    const int src_virtual = virtual_rank - mask;
    const int src_rank = (src_virtual + root) % comm_size;
    return MPI_Recv(buffer, count, datatype, src_rank, kBroadcastTag, comm, MPI_STATUS_IGNORE);
  }

  return MPI_SUCCESS;
}

int TreeBroadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int comm_size = 0;
  if (MPI_Comm_size(comm, &comm_size) != MPI_SUCCESS) {
    return MPI_ERR_COMM;
  }

  int my_rank = 0;
  if (MPI_Comm_rank(comm, &my_rank) != MPI_SUCCESS) {
    return MPI_ERR_COMM;
  }

  const int validation = ValidateBroadcastArgs(buffer, count, root, comm_size);
  if (validation != MPI_SUCCESS) {
    return validation;
  }

  int type_size = 0;
  if (MPI_Type_size(datatype, &type_size) != MPI_SUCCESS || type_size <= 0) {
    return MPI_ERR_TYPE;
  }

  const int virtual_rank = (my_rank - root + comm_size) % comm_size;

  for (int mask = 1; mask < comm_size; mask <<= 1) {
    const int status = TreeBroadcastStep(buffer, count, datatype, root, virtual_rank, mask, comm_size, comm);

    if (status != MPI_SUCCESS) {
      return status;
    }
  }

  return MPI_SUCCESS;
}

template <typename T>
bool BroadcastScalar(T *value, MPI_Datatype datatype, int root_proc, MPI_Comm comm) {
  return NalitovDBroadcast(static_cast<void *>(value), 1, datatype, root_proc, comm) == MPI_SUCCESS;
}

}  // namespace

int NalitovDBroadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  return TreeBroadcast(buffer, count, datatype, root, comm);
}

NalitovDBroadcastMPI::NalitovDBroadcastMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

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

bool NalitovDBroadcastMPI::ValidationImpl() {
  int init_flag = 0;
  MPI_Initialized(&init_flag);

  if (init_flag == 0) {
    return false;
  }

  int comm_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  const auto &input_data = GetInput();
  const bool valid_type = std::holds_alternative<std::vector<int>>(input_data.data) ||
                          std::holds_alternative<std::vector<float>>(input_data.data) ||
                          std::holds_alternative<std::vector<double>>(input_data.data);

  if (!valid_type) {
    return false;
  }

  if (input_data.root < 0 || input_data.root >= comm_size) {
    return false;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int root_proc = 0;
    if (proc_rank == 0) {
      root_proc = input_data.root;
    }

    if (!BroadcastScalar(&root_proc, MPI_INT, 0, MPI_COMM_WORLD)) {
      return false;
    }

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
  const bool is_root = (proc_rank == root_proc);

  if (is_root) {
    if (!std::holds_alternative<std::vector<T>>(input_data.data)) {
      return false;
    }
    elem_count = static_cast<int>(std::get<std::vector<T>>(input_data.data).size());
  }

  if (!BroadcastScalar(&elem_count, MPI_INT, root_proc, MPI_COMM_WORLD)) {
    return false;
  }

  auto &output_result = GetOutput();

  if (!std::holds_alternative<std::vector<T>>(output_result)) {
    output_result = InTypeVariant{std::vector<T>()};
  }

  auto &dest_buffer = std::get<std::vector<T>>(output_result);

  if (elem_count == 0) {
    dest_buffer.clear();
    return true;
  }

  if (static_cast<int>(dest_buffer.size()) != elem_count) {
    dest_buffer.resize(elem_count);
  }

  if (is_root) {
    const auto &src_buffer = std::get<std::vector<T>>(input_data.data);
    std::ranges::copy(src_buffer, dest_buffer.begin());
  }

  if (NalitovDBroadcast(dest_buffer.data(), elem_count, mpi_dtype, root_proc, MPI_COMM_WORLD) != MPI_SUCCESS) {
    return false;
  }

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
