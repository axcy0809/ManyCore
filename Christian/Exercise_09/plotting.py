import matplotlib.pyplot as plt
import numpy as np

x = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
a = [3.7e-5, 3.2e-5, 6.1e-5, 7e-5, 9.3e-5, 0.000526, 0.00485]
b = [0.000212, 0.000201, 0.000257, 0.000382, 0.002429, 0.0188, 0.2556]
c = [1.2e-5, 9e-6, 1.7e-5, 3.5e-5, 0.000352, 0.0034, 0.031]

plt.figure()
plt.loglog(x, a,'-ob',label = 'own Cuda Code')
plt.loglog(x, b,'-or',label = 'own OpenCL Code')
plt.loglog(x, c,'-og',label = 'OpenMP')




plt.title('Runtimes of Dot Product', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})
