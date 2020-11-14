import matplotlib.pyplot as plt
import numpy as np

x = [1000, 10000, 100000, 1000000]
a = [0.0063, 0.007, 0.017, 0.075]
b = [0.012, 0.014, 0.033, 0.15]
c = [0.018, 0.022, 0.05, 0.22]
d = [0.024, 0.028, 0.064, 0.29]


plt.figure()
plt.semilogx(x, a,'-ob',label = 'K = 8')
plt.semilogx(x, b,'-or',label = 'K = 16')
plt.semilogx(x, c,'-og',label = 'K = 24')
plt.semilogx(x, d,'-om',label = 'K = 32')


plt.title('Runtime of dot products', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})
