
#include <iostream>

#define VIENNACL_WITH_CUDA

#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"


int main() {

  size_t N = 1000;
  viennacl::vector<double> x = viennacl::scalar_vector<double>(N, 1.0);
  viennacl::vector<double> y = viennacl::scalar_vector<double>(N, 2.0);

  double z = viennacl::linalg::inner_prod(x+y, x-y);
  
  std::cout << "Inner Product = " << z << std::endl;

  return EXIT_SUCCESS;
}
