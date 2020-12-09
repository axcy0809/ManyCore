#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>

int main(void) {

    int N = 1000;

  // initialize x and y
  thrust::host_vector<double> h_x(N, 1.0);
  thrust::host_vector<double> h_y(N, 2.0);


  // transfer data to the device
  thrust::device_vector<double> d_x= h_x;
  thrust::device_vector<double> d_y= h_y;

  double z = thrust::inner_product(d_x+d_y, d_x-d_y);

  std::cout << "Inner Product = " << z << std::endl;

  return 0;
}
