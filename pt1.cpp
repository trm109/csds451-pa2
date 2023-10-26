#include <sycl/sycl.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
using namespace sycl;
static const int N = 16;

std::vector<float> generate_matrix(int size, int seed){
  std::vector<float> matrix(size*size);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  std::generate(matrix.begin(), matrix.end(), [&](){return dis(gen);});
  return matrix;
}
int main(){
  //# define queue which has default device associated for offload
  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  //# Unified Shared Memory Allocation enables data access on host and device
  int *data = malloc_shared<int>(N, q);

  //# Initialization
  for(int i=0; i<N; i++) data[i] = i;

  //# Offload parallel computation to device

  //# Print Output
  for(int i=0; i<N; i++) std::cout << data[i] << "\n";

  free(data, q);
  std::cout << "Completed" << std::endl;
  return 0;
}
