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
  thrust::device_vector<double> d_x = h_x;
  thrust::device_vector<double> d_y = h_y;
  thrust::device_vector<double> d_arg1;
  thrust::device_vector<double> d_arg2;


  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_arg1.begin(), thrust::plus<double>());
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_arg2.begin(), thrust::minus<double>());

  double z = thrust::inner_product(d_x.begin(), d_x.end(), d_y.begin(), 0);

  std::cout << "Inner Product = " << z << std::endl;

  return 0;
}
