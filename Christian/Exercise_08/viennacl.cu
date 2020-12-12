#include "timer.hpp"
#include <iostream>

#define VIENNACL_WITH_CUDA

#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"


int main() {

  Timer timer;
  size_t N = 1000;
  viennacl::vector<double> x = viennacl::scalar_vector<double>(N, 1.0);
  viennacl::vector<double> y = viennacl::scalar_vector<double>(N, 2.0);

  std::vector<double> timings;
  double z;
  for(int reps=0; reps < 10; ++reps) {
      timer.reset();
      z = viennacl::linalg::inner_prod(x+y, x-y);
      timings.push_back(timer.get());        
    }
  std::sort(timings.begin(), timings.end());
  double time_elapsed = timings[10/2];

  std::cout << "Time elapsed: " << time_elapsed << std::endl << std::endl;
  
  std::cout << "Inner Product = " << z << std::endl;

  return EXIT_SUCCESS;
}
