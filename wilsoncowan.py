from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from skccm.utilities import train_test_split
import skccm.data as data
from skccm import Embed
import skccm as ccm
import argparse
import logging

def sigmoid_function(x, a, theta):
    return 1 / (1 + math.exp(-a*(x - theta))) - 1 / (1 + math.exp(a*theta))


def wilson_cowan(x, t, k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P, Q):
    # assign each ODE to a vector element
    E, I = x[0], x[1]

    # define each ODE
    dEdt = (-E + (k_e - E)*sigmoid_function(c1*E - c2*I + P, a_e, theta_e))/tau_e
    dIdt = (-I + (k_i - I)*sigmoid_function(c3*E - c4*I + Q, a_i, theta_i))/tau_i

    return [dEdt, dIdt]


def coupled_wilson_cowan(x, t, alpha, beta,
                         k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P1, P, Pp, direction):
    E1, I1, E2, I2 = x[0], x[1], x[2], x[3]
    if direction == 0:
        dE1dt = (-E1 + (k_e - E1)*sigmoid_function(c1 *
                 E1 - c2*I1 + P1, a_e, theta_e))/tau_e
        dI1dt = (-I1 + (k_i - I1)*sigmoid_function(c3 *
                 E1 - c4*I1, a_i, theta_i))/tau_i
        dE2dt = (-E2 + (k_e - E2)*sigmoid_function(c1 *
                 E2 - c2*I2 + alpha*E1, a_e, theta_e))/tau_e
        dI2dt = (-I2 + (k_i - I2)*sigmoid_function(c3 *
                 E2 - c4*I2 + beta*E1, a_i, theta_i))/tau_i
    if direction == 1:
        dE1dt = (-E1 + (k_e - E1)*sigmoid_function(c1 *
                 E1 - c2*I1 + P + alpha*E2, a_e, theta_e))/tau_e
        dI1dt = (-I1 + (k_i - I1)*sigmoid_function(c3 *
                 E1 - c4*I1, a_i, theta_i))/tau_i
        dE2dt = (-E2 + (k_e - E2)*sigmoid_function(c1 *
                 E2 - c2*I2 + Pp + alpha*E1, a_e, theta_e))/tau_e
        dI2dt = (-I2 + (k_i - I2)*sigmoid_function(c3 *
                 E2 - c4*I2, a_i, theta_i))/tau_i

    return [dE1dt, dI1dt, dE2dt, dI2dt]


def power_spectrum(x, t, step):
    fourier_transform = np.fft.rfft(x-np.mean(x))
    abs_ft = np.abs(fourier_transform)
    powerspectrum = np.square(abs_ft)/len(t)
    frequency = np.linspace(0, 1/(2*step), len(powerspectrum))
    X = [frequency, powerspectrum]

    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Wilson-Cowan model analysis')
    parser.add_argument('-c', '--connectivity',
                        type=int, default=1, help='0 for unidirectional, 1 for bidirectional. The default is 1')
    parser.add_argument('-df', '--dynamics_figure', 
                        action='store_true', help='Plot dynamics characteristics')
    parser.add_argument('-mi', '--mutual_information',
                        action='store_true', help='Plot mutual information')
    parser.add_argument('-e', '--embedding',
                        action='store_true', help='Embedding')
    parser.add_argument('-ic', '--ic_sensibility',
                        action='store_true', help='Plot of initial condition sensibility')
    parser.add_argument('-ps', '--power_spectrum_',
                        action='store_true', help='Power spectrum')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    # Parameters
    P1 = 1.5
    tau_e = 1.0
    tau_i = 1.0
    k_e = 1.0
    k_i = 1.0
    c1 = 16.0
    c2 = 12.0
    c3 = 15.0
    c4 = 3.0
    theta_e = 4.0
    a_e = 1.3
    theta_i = 3.7
    a_i = 2.0
    P = 1.09
    Pp = 1.06
    if args.connectivity == 0:
        logging.info('Unidirectional coupling Wilson-Cowan circuits')
        direction = 0
    else:
        logging.info('Bidirectional coupling Wilson-Cowan circuits')
        direction = 1

    # Bifurcation parameters
    # alpha, beta = (7.0, 2.0)
    # alpha, beta = (5.8, 0.1)
    if direction == 0:
        alpha, beta = (5.3, -2.0)
        logging.info(f'alpha, beta = ({alpha}, {beta})')
    else:
        alpha, beta = (1.3, -2.0)
        logging.info(f'alpha, beta = ({alpha}, {beta})')


    # initial conditions
    x0 = [0.1, 0.1, 0.1, 0.1]
    step = 0.1
    # declare a time vector (time window)
    t = np.arange(0, 1000, step)

    x = odeint(coupled_wilson_cowan, x0, t, args=(alpha, beta,
        k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P1, P, Pp, direction))

    E1 = x[:, 0]
    I1 = x[:, 1]
    E2 = x[:, 2]
    I2 = x[:, 3]

    if args.dynamics_figure:
        fig = plt.figure()
        ax1 = plt.subplot(421)
        ax1.plot(t[:3000], E1[:3000])
        ax1.set_ylabel('E_1')
        plt.setp(ax1.get_xticklabels(), fontsize=5)
        ax1.set_title('Time Series')

        ax2 = plt.subplot(423)
        ax2.plot(t[:3000], I1[:3000])
        ax2.set_ylabel('I_1')
        plt.setp(ax2.get_xticklabels(), fontsize=5)

        ax3 = plt.subplot(425)
        ax3.plot(t[:3000], E2[:3000])
        ax3.set_ylabel('E_2')
        plt.setp(ax3.get_xticklabels(), fontsize=5)

        ax4 = plt.subplot(427)
        ax4.plot(t[:3000], I2[:3000])
        ax4.set_ylabel('I_2')
        ax4.set_xlabel('Time')
        plt.setp(ax4.get_xticklabels(), fontsize=5)

        ax5 = plt.subplot(4,2,(6,8))
        ax5.plot(E2, I2)
        ax5.set_xlabel('E_2')
        ax5.set_ylabel('I_2')
        ax5.set_title('Phase Space')
        ax5.grid()

        ax6 = plt.subplot(4, 2, (2, 4))
        ax6.plot(E1, I1)
        ax6.set_xlabel('E_1')
        ax6.set_ylabel('I_1')
        ax6.grid()
        plt.show()

    if args.ic_sensibility:
        x01 = [0.1001, 0.1001, 0.1001, 0.1001]
        x1 = odeint(coupled_wilson_cowan, x01, t, args=(alpha, beta,
                             k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P1, P, Pp, direction))
        E11 = x1[:, 0]
        I11 = x1[:, 1]
        E21 = x1[:, 2]
        I21 = x1[:, 3]

        plt.figure()
        plt.title('Initial condition sensibility')
        plt.plot(t[:3200], E21[:3200], 'k--',
                 label=f'{x01}')
        plt.plot(t[:3200], E2[:3200], 'r', alpha=0.6,
                 label=f'{x0}')
        plt.grid()
        plt.legend()
        plt.show()

    if args.power_spectrum_:
        powerspectrum = [power_spectrum(E1,t,step), power_spectrum(I1,t,step),
                            power_spectrum(E2,t,step), power_spectrum(I2,t,step)]

        ax = plt.subplot(411)
        ax.plot(powerspectrum[0][0], powerspectrum[0][1])
        ax.set_title('E_1')
        ax = plt.subplot(412)
        ax.plot(powerspectrum[1][0], powerspectrum[1][1])
        ax.set_title('I_1')
        ax = plt.subplot(413)
        ax.plot(powerspectrum[2][0], powerspectrum[2][1])
        ax.set_title('E_2')
        ax = plt.subplot(414)
        ax.plot(powerspectrum[3][0], powerspectrum[3][1])
        ax.set_xlabel('Frequency')
        ax.set_title('I_2')
        plt.show()

    e1, e2, e3, e4 = Embed(E1), Embed(I1), Embed(E2), Embed(I2)

    # Mutual Information
    if args.mutual_information:
        fig, axs = plt.subplots(4)
        axs[0].plot(e1.mutual_information(500))
        axs[0].set_title('E_1')
        axs[1].plot(e2.mutual_information(500))
        axs[1].set_title('I_1')
        axs[2].plot(e3.mutual_information(500))
        axs[2].set_title('E_2')
        axs[3].plot(e4.mutual_information(500))
        axs[3].set_title('I_2')
        axs[3].set_xlabel('Lags')
        fig.suptitle('Mutual Information')
        plt.show()

    # Set the embedding dimension and lag
    if args.embedding:
        lag = int(input('Select lag value (first minimum on MI): '))
        embed_list = np.arange(1,10,1)
        bestX2X1 = []
        bestX1X2 = []
        for embed in embed_list:
            X1 = e1.embed_vectors_1d(lag, embed)
            X2 = e3.embed_vectors_1d(lag, embed)

            # fig, axs = plt.subplots(1, 2)
            # axs[0].scatter(X1[:, 0], X1[:, 1], s=0.2)
            # axs[0].set_xlabel('E_1(t)')
            # axs[0].set_ylabel(f'E_1(t+{lag})')
            # axs[1].scatter(X2[:, 0], X2[:, 1], s=0.2)
            # axs[1].set_xlabel('E_2(t)')
            # axs[1].set_ylabel(f'E_2(t+{lag})')
            # fig.suptitle(f'Embedding with dim {embed}')

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
            bestX2X1.append(sc1[-1])
            bestX1X2.append(sc2[-1])
            # plt.figure()
            # plt.title('Prediction skill as function of library lenght')
            # plt.plot(lib_lens, sc1, label=f'X2 used to predict X1')
            # plt.plot(lib_lens, sc2, label=f'X1 used to predict X2')
            # plt.xlabel('Library lenght')
            # plt.grid()
            # plt.legend()
            # plt.title(f'Forecast skill (embedding dim {embed})')

        plt.figure()
        plt.title('Prediction skills as function of embedding dimension')
        plt.plot(embed_list, bestX2X1, label="X2 for X1")
        plt.plot(embed_list, bestX1X2, label="X1 for X2")
        plt.grid()
        plt.legend()
        plt.show()

