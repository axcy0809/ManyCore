#include <iostream>
#include <stdexcept>
#include <vexcl/vexcl.hpp>
#include "timer.hpp"


int main() {
 vex::Context ctx(vex::Filter::GPU&&vex::Filter::DoublePrecision);
 
 std::cout << ctx << std::endl; // print list of selected devices

 size_t N = 1000;
 Timer timer;
 std::vector<double> a(N, 1.0), b(N, 2.0);

 vex::vector<double> A(ctx, a);
 vex::vector<double> B(ctx, b);

std::vector<double> timings;
  double z;
  for(int reps=0; reps < 10; ++reps) {
      timer.reset();
    vex::Reductor<double, vex::SUM> sum(ctx);
    z = sum((A+B) * (A-B));
    timings.push_back(timer.get());        
    }
  std::sort(timings.begin(), timings.end());
  double time_elapsed = timings[10/2];

  std::cout << "Time elapsed: " << time_elapsed << std::endl << std::endl;

 std::cout << "Dot Product = " << z << std::endl;

 return 0;
}