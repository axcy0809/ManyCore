
import matplotlib.pyplot as plt
import numpy as np

number = 1e8

x = np.arange(1, 128, 2).tolist()
a = [0.012571,
0.017486,
0.015078,
0.012117,
0.010651,
0.009733,
0.009158,
0.008524,
0.006787,
0.006837,
0.006616,
0.008059,
0.007372,
0.007131,
0.006868,
0.006503,
0.006211,
0.00624,
0.006126,
0.005961,
0.00579,
0.005682,
0.005535,
0.005049,
0.004882,
0.004684,
0.004515,
0.004283,
0.004131,
0.003953,
0.003767,
0.003576,
0.003459,
0.003458,
0.003518,
0.003505,
0.003451,
0.003425,
0.003408,
0.003374,
0.003308,
0.003319,
0.003271,
0.003238,
0.003243,
0.003199,
0.003193,
0.003171,
0.003101,
0.003064,
0.003023,
0.002995,
0.002944,
0.002435,
0.00238,
0.002396,
0.002329,
0.002257,
0.002295,
0.00219,
0.002215,
0.002164,
0.002137,
0.002178]
b = np.true_divide(np.true_divide(number*3*8/1e9, a),x)


plt.figure()
plt.plot(x, a,'-ob',label = 'runtime strided access')


plt.title('Runtime for strided memory access', size = 25)
plt.xlabel('k [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})





plt.figure()
plt.plot(x, b,'-ob',label = 'Bandwidth')


plt.title('Bandwidth for N/k elements', size = 25)
plt.xlabel('k [-]', size=25)
plt.ylabel('bandwidth [GB/sec]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})