#include <sycl/sycl.hpp>
using namespace sycl;
static const int N = 16;

std::vector<float> generate_matrix(int size, int seed){
  std::vector<float> matrix(size*size);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      matrix[i*size+j] = dist(rng);
    }
  }
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
