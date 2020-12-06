const char *my_opencl_program = ""
    "__kernel void vec_mult(__global double *x,\n"
    "                      __global double *y,\n"
    "                      __global double *result,\n"
    "                      unsigned int N\n)"
    "{\n"
    "   int d = 0;\n"
    "  for (unsigned int i  = get_global_id(0);\n"
    "                    i  < N;\n"
    "                    i += get_global_size(0))\n"
    "{\n"
    "    result[i] = x[i] * y[i];\n"
    "}\n"
    "}"; 

// CPP code for to illustrate  
// std::string::push_back() 
   
#include <iostream> 
#include <string> 
   

int main() 
{ 
    size_t source_len = std::string(my_opencl_program).length();
    int M = 1;
    char newKernel[source_len+1+(5*M)];
    char addline[5] = {'d','=','0',';','\0'};
    //char addline[] = {'d=0;\n'};

        for(int i = 0; i < source_len-1; i++)
        {
            newKernel[i] = my_opencl_program[i];
        }
        for (int m = 0;m<M;m++)
        {
            for(int i = 0; i < 5; i++)
            {
                newKernel[source_len + i + m*5] = addline[i];
                //std::cout << addline[i] << std::endl;
                //std::cout << newKernel[source_len + i] << std::endl;
            }
        }

        newKernel[source_len+1+(5*M)] = {'}'};
        for(int i = 0; i < source_len+1+(5*M)+1; i++)
        {
            std::cout << newKernel[i] << std::endl;
        }
    
    std::cout << std::string(newKernel) << std::endl;


    return 0; 
} 