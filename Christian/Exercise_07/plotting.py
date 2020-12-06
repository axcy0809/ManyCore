import matplotlib.pyplot as plt
import numpy as np

x = [1000, 100000, 10000000, 1000000000]
a = [0.0002, 0.0022, 0.25, 2.5]
b = [4.5e-5, 0.0021, 0.22, 2.6]
c = [0.000139, 0.000688, 0.031396, 0.302954]

plt.figure()
plt.loglog(x, a,'-ob',label = 'OpenCL on GPU')
plt.loglog(x, b,'-or',label = 'OpenCL on CPU')
plt.loglog(x, d,'-og',label = 'CUDA')


plt.title('Runtimes of Dot Product', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})