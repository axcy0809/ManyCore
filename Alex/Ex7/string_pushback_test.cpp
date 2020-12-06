
// CPP code for to illustrate  
// std::string::push_back() 
   
#include <iostream> 
#include <string> 
using namespace std; 
   
// Function to demonstrate push_back() 
string diffKernel(string str1) 
{
    string addStr("j = 1;\n");

    for(int i = 0; addStr[i] != '\0'; i++) 
    { 
        str1.push_back(addStr[i]); 
    } 
    return str1;
} 
          
// Driver code 
int main() 
{ 
    string str1("        result[i] = x[i] * y[i];\n"); 
    cout << str1 << endl; 
    for (int i = 0; i < 30; i++)
    {
        str1 = diffKernel(str1);
    }

    string klam("}");
    str1.push_back(klam[0]);
    cout << str1 << endl;
   
    return 0; 
} 
