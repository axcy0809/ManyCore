import matplotlib.pyplot as plt
import numpy as np

x = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
a = [0.000244, 0.00024, 0.000235, 0.000394, 0.002764, 0.025515, 0.029741]
b = [0.000248, 0.000251, 0.000247, 0.000416, 0.002925, 0.0272, 0.0354]
c = [0.000242, 0.000246, 0.00024, 0.0004, 0.00281, 0.0256, 0.02822]


plt.figure()
plt.loglog(x, a,'-ob',label = 'exclusive')
plt.loglog(x, b,'-or',label = 'inclusive reusing implementation')
plt.loglog(x, c,'-og',label = 'inclusive modified')


plt.title('Runtime Scans', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})