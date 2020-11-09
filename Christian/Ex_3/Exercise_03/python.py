import matplotlib.pyplot as plt
import numpy as np

number = 1e8

x = np.arange(0, 64, 1).tolist()
a = [0.010112,
0.012768,
0.012712,
0.012762,
0.01338,
0.013418,
0.013384,
0.013459,
0.013532,
0.013439,
0.013353,
0.01332,
0.013186,
0.013223,
0.013164,
0.013241,
0.011236,
0.012968,
0.01299,
0.012939,
0.013191,
0.013275,
0.013246,
0.013303,
0.013357,
0.013485,
0.013422,
0.013494,
0.013455,
0.013031,
0.012989,
0.012985,
0.010154,
0.012694,
0.012762,
0.012694,
0.013237,
0.013327,
0.013335,
0.013304,
0.013424,
0.01334,
0.013301,
0.01334,
0.013155,
0.013184,
0.013125,
0.013156,
0.011188,
0.012844,
0.012904,
0.012892,
0.013065,
0.01324,
0.01324,
0.013197,
0.013248,
0.013444,
0.013371,
0.013383,
0.013337,
0.012981,
0.013027,
0.013029]

temp = np.empty(64)
temp.fill(number)
b = np.true_divide((temp-x)*8*3/1e9,a)


plt.figure()
plt.plot(x, a,'-ob',label = 'runtime offset access')


plt.title('Runtime for offset memory access', size = 25)
plt.xlabel('k [-]', size=25)
plt.ylabel('runtime [s]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})





plt.figure()
plt.plot(x, b,'-ob',label = 'Bandwidth')


plt.title('Bandwidth for N-k elements', size = 25)
plt.xlabel('k [-]', size=25)
plt.ylabel('bandwidth [GB/sec]', size = 25)
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25)
plt.grid()
plt.legend(prop={'size': 20})
