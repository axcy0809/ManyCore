import matplotlib.pyplot as plt
import numpy as np

x = [1000, 100000, 10000000, 1000000000]
a = [3.4e-5, 7.4e-5, 0.0043, 0.068]
b = [6.7e-5, 0.000135, 0.0074, 0.0585]
c = [4.1e-5, 8.1e-5, 0.004651, 0.0377]
d = [2.5e-5, 4.6e-5, 0.0024, 0.023822]


plt.figure()
plt.loglog(x, a,'-ob',label = 'shuffled')
plt.loglog(x, b,'-or',label = 'shared')
plt.loglog(x, d,'-og',label = 'dot product shuffled')
plt.loglog(x, c,'-om',label = 'dot product shared')



plt.title('Runtime calculations', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})