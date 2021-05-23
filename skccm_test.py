from skccm.utilities import train_test_split
import skccm.data as data
from skccm import Embed
import skccm as ccm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from mpl_toolkits import mplot3d

def lorenz(v, t, sigma=10., beta=8./3, rho=28.0):
    x = v[0]
    y = v[1]
    z = v[2]
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

if __name__ == "__main__":
    logistic = False
    lorenz_ = True
    if logistic:
        # LOGISTIC
        rx1 = 3.72  # determines chaotic behavior of the x1 series
        rx2 = 3.72  # determines chaotic behavior of the x2 series
        b12 = 0.2  # Influence of x1 on x2
        b21 = 0.01  # Influence of x2 on x1
        ts_length = 1000
        x1, x2 = data.coupled_logistic(rx1, rx2, b12, b21, ts_length)

        fig = plt.figure()
        ax = [plt.subplot(212), plt.subplot(211)]
        ax[0].plot(x1[:100])
        ax[1].plot(x2[:100])
        fig.suptitle('Logistic Dynamics')


        lag = 1
        embed = 2
        e1 = Embed(x1)
        e2 = Embed(x2)
        X1 = e1.embed_vectors_1d(lag, embed)
        X2 = e2.embed_vectors_1d(lag, embed)
        print(X1.shape, X2.shape)

        # Embedding dimension
        fig, axs = plt.subplots(1,2)
        axs[0].scatter(X1[:,0], X1[:,1], s=0.2)
        axs[1].scatter(X2[:,0], X2[:,1], s=0.2)
        fig.suptitle('Embedding')


        #split the embedded time series
        x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)

        CCM = ccm.CCM()

        #library lengths to test
        len_tr = len(x1tr)
        lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')

        #test causation
        CCM.fit(x1tr, x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)

        sc1, sc2 = CCM.score()
        plt.figure()
        plt.plot(lib_lens, sc1, label='score X1')
        plt.plot(lib_lens, sc2, label='score X2')
        plt.grid()
        plt.legend()
        plt.title('Forecast skill')

    if lorenz_:
        # LORENZ
        X = odeint(lorenz, [1,1,1], np.linspace(0,100,10000))

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X[:, 0], X[:, 1], X[:, 2])
        plt.title('Lorenz Attractor')

        # Time Series
        fig = plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(X[:, 0])
        ax2 = plt.subplot(312)
        ax2.plot(X[:, 1])
        ax3 = plt.subplot(313)
        ax3.plot(X[:, 2])
        fig.suptitle('Time Series')

        e1 = Embed(X[:,0])
        e2 = Embed(X[:,1])
        e3 = Embed(X[:,2])

        fourier_transform = np.fft.rfft(X[:, 0]-np.mean(X[:, 0]))
        abs_ft = np.abs(fourier_transform)
        powerspectrum = np.square(abs_ft)/len(np.linspace(0, 100, 10000))
        frequency = np.linspace(0, 1/(2*0.01), len(powerspectrum))

        plt.figure('PS')
        plt.plot(frequency, powerspectrum)
        plt.yscale('log')
        plt.grid()


        # Mutual Information
        fig, axs = plt.subplots(3)
        axs[0].plot(e1.mutual_information(100))
        axs[1].plot(e2.mutual_information(100))
        axs[2].plot(e3.mutual_information(100))
        fig.suptitle('Mutual Information')
        plt.show()



