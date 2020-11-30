import matplotlib.pyplot as plt
import numpy as np

x = [100, 10000, 1000000]
a = [0.00034, 0.000567, 0.015646]
b = [0.00014, 0.000349, 0.0134]
c = [0.000142, 0.000349, 0.0134]



plt.figure()
plt.loglog(x, a,'-ob',label = 'matrix*vector K times')
plt.loglog(x, b,'-or',label = 'Row major')
plt.loglog(x, c,'-og',label = 'Column major')




plt.title('Runtime calculations K = 10', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})














x = [100, 10000, 100000]
a = [0.029, 0.041, 0.22]
b = [0.00126, 0.014, 0.13]
c = [0.00128, 0.014, 0.13]



plt.figure()
plt.loglog(x, a,'-ob',label = 'matrix*vector K times')
plt.loglog(x, b,'-or',label = 'Row major')
plt.loglog(x, c,'-og',label = 'Column major')




plt.title('Runtime calculations K = 1000', size = 25)
plt.xlabel('N [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})
