#include <omp.h>
#include <cmath>
#include <iostream>
#include "timer.hpp"
#include <vector>
#include <algorithm>

int main()
{
Timer timer;

   int N = 100;
   double *x = (double*)malloc(sizeof(double) * N);
   double *y = (double*)malloc(sizeof(double) * N);

   for (size_t i=0; i<N; ++i) {
       x[i] = 1;
       y[i] = 2;
   }

   double dot;
   
std::vector<double> timings;
for(int reps=0; reps < 10; ++reps) {
timer.reset();
   dot = 0;

   #pragma omp target teams distribute parallel for map(to: x[0:N], y[0:N]) map(tofrom: dot) reduction(+:dot)
   for (int idx = 0; idx < N; ++idx)
   {
       dot += (x[idx] + y[idx]) * (x[idx] - y[idx]);
   } 

timings.push_back(timer.get());
}

std::sort(timings.begin(), timings.end());
double time_elapsed = timings[10/2];

std::cout << "Time elapsed: " << time_elapsed << std::endl << std::endl;

   std::cout << "Reduction result: " << dot << std::endl;
   std::cout << "Expected result: " << (-3)*N << std::endl;

   return EXIT_SUCCESS;
}