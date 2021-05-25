from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from skccm.utilities import train_test_split
from statsmodels.graphics.tsaplots import plot_acf
import skccm.data as data
from skccm import Embed
import skccm as ccm
import sdeint
import argparse
import logging

def sigmoid_function(x, a, theta):
    '''
    Sigmoid function that must be used in the Wilson-Cowan model.
    '''

    return 1 / (1 + math.exp(-a*(x - theta))) - 1 / (1 + math.exp(a*theta))


def wilson_cowan(x, t, k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P, Q):
    '''
    A simple Wilson-Cowan model.
    '''
    E, I = x[0], x[1]

    dEdt = (-E + (k_e - E)*sigmoid_function(c1*E - c2*I + P, a_e, theta_e))/tau_e
    dIdt = (-I + (k_i - I)*sigmoid_function(c3*E - c4*I + Q, a_i, theta_i))/tau_i

    return [dEdt, dIdt]


def coupled_wilson_cowan(x, t, alpha, beta,
                         k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P1, P, Pp, direction):
    '''
    Equations for two coupled Wilson-Cowan models. Direction 0 refers to the unidirectional case in which
    the first WC system is the driving force of the second one, while direcion 1 represents the bidirectional
    case in which both the excitatory populations are coupled one with the other.
    '''

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
    '''
    Power spectrum calculated using Fast Fourier Transform of Numpy. It is normalized by the lenght
    of the time series. On x-axis frequency has steps equal to the maximal frequency you can have
    if your data has been sampled at time steps equal to the step argument.
    '''
    fourier_transform = np.fft.rfft(x-np.mean(x))
    abs_ft = np.abs(fourier_transform)
    powerspectrum = np.square(abs_ft)/len(t)
    frequency = np.linspace(0, 1/(2*step), len(powerspectrum))
    X = [frequency, powerspectrum]

    return X


def autocorrelation(x, max_lag):
    try:
        from varname import nameof
    except ModuleNotFoundError:
        logging.error("No module named 'nameof'. Install it by 'pip install -U varname'.")

    x_mean = np.mean(x)
    auto_corr = []
    for d in range(max_lag):
        ac = 0
        for i in range(len(x)-d):
            ac += (x[i] - x_mean) * (x[i+d] - x_mean)
        ac = ac / np.sqrt(np.sum((x - x_mean)**2) * np.sum((x - x_mean)**2))
        auto_corr.append(ac)

    return auto_corr


def prediction_skill(x, y, lag, embed):
    ''' This function relies on skccm.Embed module. Prediction skill is represented by the coefficient
    of determination'''
    x_emb, y_emb = Embed(x), Embed(y)
    x_emb = x_emb.embed_vectors_1d(lag, embed)
    y_emb = y_emb.embed_vectors_1d(lag, embed)

    x1tr, x1te, x2tr, x2te = train_test_split(x_emb, y_emb, percent=.75)

    CCM = ccm.CCM()
    len_tr = len(x1tr)
    lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')
    CCM.fit(x1tr, x2tr)
    _, _ = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
    sc1, sc2 = CCM.score()
    return sc1, sc2, x_emb, y_emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Wilson-Cowan model analysis.')
    parser.add_argument('-c', '--connectivity',
                        type=int, default=1, help='0 for unidirectional, 1 for bidirectional. The default is 1')
    parser.add_argument('-df', '--dynamics_figure', 
                        action='store_true', help='Plot dynamics characteristics.')
    parser.add_argument('-mi', '--mutual_information',
                        action='store_true', help='Plot mutual information.')
    parser.add_argument('-ac', '--autocorr',
                        action='store_true', help='Plot autocorrelation function')
    parser.add_argument('-ic', '--ic_sensibility',
                        action='store_true', help='Plot of initial condition sensibility')
    parser.add_argument('-ps', '--power_spectrum_',
                        action='store_true', help='Power spectrum')
    parser.add_argument('-e', '--embedding',
                        action='store_true', help='Embedding')
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
        logging.info(f'Chaotic regime --> alpha, beta = ({alpha}, {beta})')
    else:
        alpha, beta = (1.3, -2.0)
        logging.info(f'Chaoitc regime --> alpha, beta = ({alpha}, {beta})')

    # initial conditions, integration step and time window
    x0 = [0.1, 0.1, 0.1, 0.1]
    step = 0.1
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
        ax1.plot(t[:3000], E1[:3000], lw=0.2)
        ax1.set_ylabel('E1')
        plt.setp(ax1.get_xticklabels(), fontsize=5)
        ax1.set_title('Time Series')

        ax2 = plt.subplot(423)
        ax2.plot(t[:3000], I1[:3000], lw=0.2)
        ax2.set_ylabel('I1')
        plt.setp(ax2.get_xticklabels(), fontsize=5)

        ax3 = plt.subplot(425)
        ax3.plot(t[:3000], E2[:3000], lw=0.2)
        ax3.set_ylabel('E2')
        plt.setp(ax3.get_xticklabels(), fontsize=5)

        ax4 = plt.subplot(427)
        ax4.plot(t[:3000], I2[:3000], lw=0.2)
        ax4.set_ylabel('I2')
        ax4.set_xlabel('Time')
        plt.setp(ax4.get_xticklabels(), fontsize=5)

        ax5 = plt.subplot(4,2,(6,8))
        ax5.plot(E2, I2, lw=0.2)
        ax5.set_xlabel('E2')
        ax5.set_ylabel('I2')
        ax5.set_title('2 dim Phase Space')
        ax5.grid()

        ax6 = plt.subplot(4, 2, (2, 4))
        ax6.plot(E1, I1, lw=0.2)
        ax6.set_xlabel('E1')
        ax6.set_ylabel('I1')
        ax6.grid()
        plt.show()

        plt.subplot(projection='3d')
        plt.plot(E1, E2, I2, lw=0.2)
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
        ax.set_title('E1')
        ax = plt.subplot(412)
        ax.plot(powerspectrum[1][0], powerspectrum[1][1])
        ax.set_title('I1')
        ax = plt.subplot(413)
        ax.plot(powerspectrum[2][0], powerspectrum[2][1])
        ax.set_title('E2')
        ax = plt.subplot(414)
        ax.plot(powerspectrum[3][0], powerspectrum[3][1])
        ax.set_xlabel('Frequency')
        ax.set_title('I2')
        plt.show()

    e1, e2, e3, e4 = Embed(E1), Embed(I1), Embed(E2), Embed(I2)

    # Mutual Information. It is calculated from skccm.Embed.mutual_information that relies of sklearn
    if args.mutual_information:
        fig, axs = plt.subplots(4)
        lag = 200
        axs[0].plot(e1.mutual_information(lag))
        axs[0].set_title('E1')
        axs[1].plot(e2.mutual_information(lag))
        axs[1].set_title('I1')
        axs[2].plot(e3.mutual_information(lag))
        axs[2].set_title('E2')
        axs[3].plot(e4.mutual_information(lag))
        axs[3].set_title('I2')
        axs[3].set_xlabel('Lags')
        fig.suptitle('Mutual Information')
        plt.show()

    if args.autocorr:
        # Maybe I can implement a test significance
        lag = 200
        auto_corrE1 = autocorrelation(E1, lag)
        auto_corrI1 = autocorrelation(I1, lag)
        auto_corrE2 = autocorrelation(E2, lag)
        auto_corrI2 = autocorrelation(I2, lag)
        plt.figure()
        plt.title('Autocorrelation')
        plt.plot(auto_corrE1, label='E1')
        plt.plot(auto_corrI1, label='I1')
        plt.plot(auto_corrE2, label='E2')
        plt.plot(auto_corrI2, label='I2')
        plt.grid()
        plt.legend()
        plt.show()


    # Set the embedding dimension and lag. Lag is choosen by looking at the first minimum in MI. It
    # represents the first time delay at which information is forgotten. Another way to select the lag
    # can be looking at the autocorrelation function, but it can give misleading results since the system
    # is not linear.
    if args.embedding:
        lag = int(input('Select lag value (first minimum on MI): '))
        embed_list = np.arange(1,10,1)
        bestE2E1, bestE1E2 = [], []
        bestI1I2, bestI2I1 = [], []
        bestE1I1, bestI1E1 = [], []
        bestE2I1, bestI1E2 = [], []
        bestE1I2, bestI2E1 = [], []
        bestE2I2, bestI2E2 = [], []
        for embed in embed_list:
            sc1ee, sc2ee, E1_emb, E2_emb = prediction_skill(E1, E2, lag, embed)
            bestE2E1.append(sc1ee[-1])
            bestE1E2.append(sc2ee[-1])

            sc1ei, sc2ei, E1_emb, I1_emb = prediction_skill(E1, I1, lag, embed)
            bestI1E1.append(sc1ei[-1])
            bestE1I1.append(sc2ei[-1])

            sc1ei1, sc2ei1, E1_emb, I2_emb = prediction_skill(E1, I2, lag, embed)
            bestI2E1.append(sc1ei1[-1])
            bestE1I2.append(sc2ei1[-1])

            sc1ei2, sc2ei2, E2_emb, I1_emb = prediction_skill(E2, I1, lag, embed)
            bestI1E2.append(sc1ei2[-1])
            bestE2I1.append(sc2ei2[-1])

            sc1ei3, sc2ei3, E2_emb, I2_emb = prediction_skill(E2, I2, lag, embed)
            bestI2E2.append(sc1ei3[-1])
            bestE2I2.append(sc2ei3[-1])

            sc1ii, sc2ii, E2_emb, I1_emb = prediction_skill(I1, I2, lag, embed)
            bestI2I1.append(sc1ii[-1])
            bestI1I2.append(sc2ii[-1])

        plt.figure()
        plt.title('Prediction skills as function of embedding dimension')
        plt.plot(embed_list, bestE2E1, label="E2 for E1")
        plt.plot(embed_list, bestE1E2, label="E1 for E2")
        plt.plot(embed_list, bestI1I2, label="I1 for I2")
        plt.plot(embed_list, bestI2I1, label="I2 for I1")
        plt.plot(embed_list, bestE1I1, label="E1 for I1")
        plt.plot(embed_list, bestI1E1, label="I1 for E1")
        plt.plot(embed_list, bestE2I1, label="E2 for I1")
        plt.plot(embed_list, bestI1E2, label="I1 for E2")
        plt.plot(embed_list, bestE1I2, label="E1 for I2")
        plt.plot(embed_list, bestI2E1, label="I2 for E1")
        plt.plot(embed_list, bestI2E2, label="I2 for E2")
        plt.plot(embed_list, bestE2I2, label="E2 for I2")
        plt.ylabel('Coefficient of determination')
        plt.grid()
        plt.legend()
        plt.show()

        embed = int(input('Choose dimension of the embedding space: '))
        sc1ee, sc2ee, E1emb, E2emb = prediction_skill(E1, E2, lag, embed)
        sc1ii, sc2ii, I1emb, I2emb = prediction_skill(I1, I2, lag, embed)

        fig = plt.figure()
        plt.title(f'Embedded attractors. Dimension equal to {embed}')
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter(E1emb[:, 0], E1emb[:, 1], E1emb[:, 2], s=0.2)
        ax.set_xlabel('E1(t)')
        ax.set_ylabel(f'E1(t+{lag})')
        ax.set_zlabel(f'E1(t+2*{lag})')

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.scatter(E2emb[:, 0], E2emb[:, 1], E2emb[:, 2], s=0.2)
        ax.set_xlabel('E2(t)')
        ax.set_ylabel(f'E2(t+{lag})')
        ax.set_zlabel(f'E2(t+2*{lag})')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(I1emb[:, 0], I1emb[:, 1], I1emb[:, 2], s=0.2)
        ax.set_xlabel('I1(t)')
        ax.set_ylabel(f'I1(t+{lag})')
        ax.set_zlabel(f'I1(t+2*{lag})')

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(I2emb[:, 0], I2emb[:, 1], I2emb[:, 2], s=0.2)
        ax.set_xlabel('I2(t)')
        ax.set_ylabel(f'I2(t+{lag})')
        ax.set_zlabel(f'I2(t+2*{lag})')

        plt.figure()
        plt.title('Prediction skill as function of library lenght')
        plt.plot(sc1ee, label=f'E2 used to predict E1')
        plt.plot(sc2ee, label=f'E1 used to predict E2')
        plt.plot(sc1ii, label=f'I1 used to predict I2')
        plt.plot(sc2ii, label=f'I2 used to predict I1')
        plt.xlabel('Library lenght')
        plt.grid()
        plt.legend()
        plt.show()

