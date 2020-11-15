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



# part 2

x = [10000, 1000000, 4000000, 4410000]
a = [0.00067, 0.0013, 0.0046]
b = [158, 1632, 3296]
c = [0.00065, 0.0012, 0.0038, 0.0043]
d = [158, 1632, 3296, 3463]



plt.figure()
plt.loglog(x[:-1], a,'-ob',label = 'runtime standard')
plt.loglog(x, c,'-og',label = 'runtime pipelined')



plt.title('Runtime of cg', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})



plt.figure()
plt.loglog(x[:-1], b,'-or',label = 'number iterations')
plt.loglog(x, d,'-om',label = 'number iterations')


plt.title('Number of iterations cg', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('iterations [-]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})
