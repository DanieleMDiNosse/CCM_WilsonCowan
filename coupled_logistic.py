import numpy as np
import matplotlib.pyplot as plt
from wilsoncowan import granger_causality_test
from initial import crosscorr

time = np.arange(0,500,0.1)
x, y = np.empty(len(time)), np.empty(len(time))
x[0], y[0] = 0.2, 0.4
beta = 0.02
for t in range(len(time)-1):
    x[t+1] = 3.9*x[t]*(1 - x[t] - beta*y[t])
    y[t+1] = 3.7*y[t]*(1 - y[t] - 0.2*x[t])

plt.figure()
plt.plot(time, x, 'k', label='x', lw=1.2)
plt.plot(time, y, 'g', label='y', lw=1.2)
plt.legend()
plt.xlabel('Time')
plt.grid()
plt.show()

crosscorr(x,y, 500)
plt.show()


for t in range(len(time)-1):
    x[t+1] = 3.9*x[t]*(1 - x[t] - 0.02*y[t])
    y[t+1] = 3.7*y[t]*(1 - y[t] - 0.2*x[t])
print(f'\n {beta} ========= Y Granger causes X =========')
granger_causality_test(x, y, [2], verbose=True)
print(f'\n {beta} ========= X Granger causes Y =========')
granger_causality_test(y, x, [2], verbose=True)
