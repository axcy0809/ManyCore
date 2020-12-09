 #include <vector>
    #include <algorithm>
    #include <iostream>
     
    #include <boost/compute/algorithm/transform.hpp>
    #include <boost/compute/container/vector.hpp>
    #include <boost/compute/functional/math.hpp>
    #include <boost/geometry/arithmetic/dot_product.hpp>
    #include <boost/range/numeric.hpp>
     
    namespace compute = boost::compute;
     
    int main()
    {
        int N = 1000;

        // get default device and setup context
        compute::device device = compute::system::default_device();
        compute::context context(device);
        compute::command_queue queue(context, device);
     
        // generate random data on the host
        std::vector<double> x(N, 1);
        std::vector<double> y(N, 2);
     
       // create a vector on the device
       compute::vector<double> d_x(x.size(), context);
       compute::vector<double> d_y(y.size(), context);
     
       // transfer data from the host to the device
       compute::copy(x.begin(), x.end(), d_x.begin(), queue);
       compute::copy(y.begin(), y.end(), d_y.begin(), queue);

        // calculate inner product
        double z = boost::inner_product((d_x + d_y), (d_x - d_y), 0);
        
        std::cout << "Dot Product = " << z << std::endl; 
     
        return 0;
    }