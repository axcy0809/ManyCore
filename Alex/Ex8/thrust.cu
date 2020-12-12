#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/for_each.h>
#include <cstdlib>
#include <vector>

typedef double       ScalarType;

int main(void) {
/*
  // generate 32M random numbers on the host
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec= h_vec;
  // sort data on the device
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  std::cout << h_vec[0] << ", " << h_vec[1] << ", " << h_vec[2] << std::endl;
*/
  std::vector<double> v1(100,1);
  std::vector<double> v2(100,2);
  std::vector<double> v1pv2(100,1);
  std::vector<double> v1mv2(100,2);

  
  for (int i = 0; i < 100; i++)
  {
        v1pv2[i] = v1[i] + v2[i];
	v1mv2[i] = v1[i] - v2[i];
  }
  

  std::cout << "v1pv2: " << v1pv2[0] << std::endl;
  std::cout << "v1mv2: " << v1mv2[0] << std::endl;

  thrust::host_vector<double> h_v1 = v1pv2;
  thrust::host_vector<double> h_v2 = v1mv2;

  //h_v1 = h_v1 + h_v2;
  //h_v2 = h_v1 - h_v2;

  //thrust::generate(h_v1.begin(), h_v1.end(), rand);
  //thrust::generate(h_v2.begin(), h_v2.end(), rand);

  thrust::device_vector<ScalarType> d_v1 = h_v1;
  thrust::device_vector<ScalarType> d_v2 = h_v2;

  ScalarType start = 0;

  //ScalarType expected_thrust = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), start);
  ScalarType expected_thrust = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), start);
  std::cout << "host dot: " << expected_thrust << std::endl;

  return 0;
}
