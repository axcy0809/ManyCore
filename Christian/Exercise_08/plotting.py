import matplotlib.pyplot as plt
import numpy as np

x = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
a = [3.7e-5, 3.2e-5, 6.1e-5, 7e-5, 9.3e-5, 0.000526, 0.00485]
b = [0.000212, 0.000201, 0.000257, 0.000382, 0.002429, 0.0188, 0.2556]
c = [0.000151, 0.000144, 0.000139, 0.000136, 0.000459, 0.002374, 0.0157]
d = [0.00025, 0.000312, 0.000158, 0.000325, 0.000405, 0.000765, 0.001285]
e = [0.000971, 0.00653, 0.069, 0.67, ]
f = [0.0001, 8.7e-5, 0.000128, 0.000127, 0.000166, 0.000784, 0.005]

plt.figure()
plt.loglog(x, a,'-ob',label = 'own Cuda Code')
plt.loglog(x, b,'-or',label = 'own OpenCL Code')
plt.loglog(x, c,'-og',label = 'viennacl')
plt.loglog(x, d,'-om',label = 'vex')
plt.loglog(x[0:4], e,'-oy',label = 'boost')
plt.loglog(x, f,'-oc',label = 'thrust')



plt.title('Runtimes of Dot Product', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})