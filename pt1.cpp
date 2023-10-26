#include <sycl/sycl.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
using namespace sycl;
static const int N = 2048;
static const int N_workers = N / 2;

std::vector<float> generate_matrix(int size, int seed){
  std::vector<float> matrix(size*size);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  std::generate(matrix.begin(), matrix.end(), [&](){return dis(gen);});
  return matrix;
}
void print_matrix_chunk(std::vector<float> matrix, int size, int chunk_size){
  for (int i = 0; i < chunk_size; i++) {
    for (int j = 0; j < chunk_size; j++){
      std::cout << matrix[i*size+j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(){
  //# print the size and worker size.
  std::cout << "Matrix size: " << N << std::endl;
  std::cout << "Worker size: " << N_workers << std::endl;
  //# define queue which has default device associated for offload
  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  //# Unified Shared Memory Allocation enables data access on host and device
  int *data = malloc_shared<int>(N, q);

  //# Initialization of input matricies.
  std::vector<float> A = generate_matrix(N, 1);
  std::vector<float> B = generate_matrix(N, 2);
  //# Create output matrix.
  std::vector<float> C(N*N);
  //# Create buffers.
  buffer bufA(A);
  buffer bufB(B);
  buffer bufC(C);
  //# Offload parallel computation to device
  q.submit([&](handler &h){
    //# Create accessors.
    accessor aA(bufA, h, read_only);
    accessor aB(bufB, h, read_only);
    accessor aC(bufC, h, write_only);
    
    //# Define size.
    range<2> global_size(N, N);
    range<2> work_group_size(N_workers, N_workers);
    //# Parallel computation.
    h.parallel_for(nd_range<2>(global_size, work_group_size), [=](nd_item<2> it){
      //# Get global id.
      int i = it.get_global_id(0);
      int j = it.get_global_id(1);
      //# Compute from local mem.
      float sum = 0.f;
      for (int k = 0; k < N; k++) {
        sum += aA[i*N+k] * aB[k*N+j];
      }
      //# Store to global mem.
      aC[i*N+j] = sum;
    });
  });
  host_accessor hC(bufC, read_only);
  //# Print Output
  std::cout << "Output Matrix" << std::endl;
  print_matrix_chunk(C, N, 4);
  free(data, q);
  std::cout << "Completed" << std::endl;
  return 0;
}
