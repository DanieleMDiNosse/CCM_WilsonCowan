from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from skccm.utilities import train_test_split
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import skccm.data as data
from skccm import Embed
import skccm as ccm
import sdeint
import argparse
import logging
from tqdm import tqdm
from pyinform import mutual_info
from sklearn.metrics import mutual_info_score

def sigmoid_function(x, a, theta):
    '''
    Sigmoid function that must be used in the Wilson-Cowan model.
    '''

    return 1 / (1 + np.exp(-a*(x - theta))) - 1 / (1 + np.exp(a*theta))


def wilson_cowan(x, t, k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, a_e, a_i, theta_e, theta_i, P, Q):
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


def noisy_coupled_wc(alpha, beta,
                     k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P1, P, Pp, direction, dt, time, x0):


    E1 = np.zeros(len(time))
    I1 = np.zeros(len(time))
    E2 = np.zeros(len(time))
    I2 = np.zeros(len(time))
    v = np.ones(len(time))
    u = np.ones(len(time))
    [E1[0], I1[0], E2[0], I2[0]] = x0

    for t in range(len(time)-1):
        if direction == 0:
            v[t+1] = v[t] + 0.5*(1.5 - v[t])*dt + 0.2*np.sqrt(dt)*np.random.normal()
            E1[t+1] = E1[t] + (-E1[t] + (k_e - E1[t])*sigmoid_function(c1 *
                    E1[t] - c2*I1[t] + v[t], a_e, theta_e))*dt / tau_e
            I1[t+1] = I1[t] + (-I1[t] + (k_i - I1[t])*sigmoid_function(c3 *
                    E1[t] - c4*I1[t], a_i, theta_i))*dt / tau_i
            E2[t+1] = E2[t] + (-E2[t] + (k_e - E2[t])*sigmoid_function(c1 *
                    E2[t] - c2*I2[t] + alpha*E1[t], a_e, theta_e))*dt / tau_e
            I2[t+1] = I2[t] + (-I2[t] + (k_i - I2[t])*sigmoid_function(c3 *
                        E2[t] - c4*I2[t] + beta*E1[t], a_i, theta_i))*dt / tau_i
        else:
            v[t+1] = v[t] + 1*(1.5 - v[t])*dt + 0.1*np.sqrt(dt)*np.random.normal()
            u[t+1] = u[t] + 1*(1.5 - u[t])*dt + 0.1*np.sqrt(dt)*np.random.normal()
            E1[t+1] = E1[t] + (-E1[t]*dt + (k_e - E1[t])*sigmoid_function(c1 *
                    E1[t] - c2*I1[t] + v[t] + alpha*E2[t], a_e, theta_e)*dt) / tau_e
            I1[t+1] = I1[t] + (-I1[t]*dt + (k_i - I1[t])*sigmoid_function(c3 *
                    E1[t] - c4*I1[t], a_i, theta_i)*dt) / tau_i
            E2[t+1] = E2[t] + (-E2[t]*dt + (k_e - E2[t])*sigmoid_function(c1 *
                    E2[t] - c2*I2[t] + u[t] + alpha*E1[t], a_e, theta_e)*dt) / tau_e
            I2[t+1] = I2[t] +(-I2[t]*dt + (k_i - I2[t])*sigmoid_function(c3 *
                        E2[t] - c4*I2[t], a_i, theta_i)*dt) / tau_i

    return E1, I1, E2, I2, v


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


def crosscorr(x, y, max_lag, bootstrap_test=False):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cross_corr = []
    for d in range(max_lag):
        cc = 0
        for i in range(len(x)-d):
            cc += (x[i] - x_mean) * (y[i+d] - y_mean)
        cc = cc / np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        cross_corr.append(cc)
    plt.plot(cross_corr,'k')
    plt.title('Cross-correlation function')
    plt.xlabel('Lags')


    if bootstrap_test:
        cross_corr_s = []
        for i in range(100):
            xs, ys = x, y
            np.random.shuffle(xs)
            np.random.shuffle(ys)
            xs_mean = np.mean(xs)
            ys_mean = np.mean(ys)
            cross_corr_s_i = []
            for d in range(max_lag):
                cc = 0
                for i in range(len(x)-d):
                    cc += (xs[i] - xs_mean) * (ys[i+d] - ys_mean)
                cc = cc / np.sqrt(np.sum((xs - xs_mean)**2) * np.sum((ys- ys_mean)**2))
                cross_corr_s_i.append(cc)
            cross_corr_s.append(cross_corr_s_i)
        meancc = np.mean(np.array(cross_corr_s), axis=0)
        stdcc = np.std(np.array(cross_corr_s), axis=0)
        plt.plot(meancc - 3*stdcc, 'crimson', lw=0.5)
        plt.plot(meancc, 'crimson', lw=0.5)
        plt.plot(meancc + 3*stdcc, 'crimson', lw=0.5)
        plt.fill_between(np.arange(0, max_lag), meancc + 3*stdcc, meancc - 3*stdcc, color='crimson', alpha=0.6)
    plt.grid()
    return cross_corr


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
    return sc1, sc2, x_emb, y_emb, lib_lens


def granger_causality_test(x, y, maxlag, verbose=False):
    '''
    Test to check if y Granger causes x
    '''
    if type(x) != 'numpy.ndarry':
        x = np.array(x)
    if type(y) != 'numpy.ndarry':
        y = np.array(y)

    gca_list = np.array([x, y])
    gca_matrix = np.transpose(gca_list)
    gca = grangercausalitytests(gca_matrix, maxlag=maxlag, verbose=verbose)

    return gca


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Wilson-Cowan model analysis.')
    parser.add_argument('-s', '--singlewc',
                        action='store_true', help='Basic Wilson-Cowan model')
    parser.add_argument('-c', '--connectivity',
                        type=int, default=1, help='0 for unidirectional, 1 for bidirectional. The default is 1')
    parser.add_argument('-df', '--dynamics_figure', 
                        action='store_true', help='Plot dynamics characteristics.')
    parser.add_argument('-mi', '--mutual_information',
                        action='store_true', help='Plot mutual information as function on lag.')
    parser.add_argument('-ac', '--autocorr',
                        action='store_true', help='Plot autocorrelation function')
    parser.add_argument('-ic', '--ic_sensibility',
                        action='store_true', help='Plot of initial condition sensibility')
    parser.add_argument('-ps', '--power_spectrum_',
                        action='store_true', help='Power spectrum')
    parser.add_argument('-e', '--embedding',
                        action='store_true', help='Embedding')
    parser.add_argument('-gc', '--granger_causality',
                        action='store_true', help='Granger causality test')
    parser.add_argument('-f', '--free_parameters',
                        action='store_true', help='Use this parser if you wanto to choose other alpha and beta')
    parser.add_argument('-cc', '--cross_correlation',
                        action='store_true', help='Cross correlation')
    parser.add_argument('-n', '--noisywc',
                        action='store_true', help='Add noise to the first oscillator')
    parser.add_argument('-mnr', '--multiple_noisy_run',
                        action='store_true', help='Solve 30 times WC noisy model in order to plot mean and std in the embedding part')

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
    Q = -0.65
    Pp = 1.06

    if args.singlewc:
        P = 1.5
        Q = -1
        x0 = [0.1, 0.1]
        step = 0.01
        t = np.arange(0,100, step)
        x = odeint(wilson_cowan, x0, t, args=(k_e, k_i, c1, c2, c3, c4, 
                            tau_e, tau_i, a_e, a_i, theta_e, theta_i, P, Q))
        E, I = x[:, 0], x[:, 1]
        fig = plt.figure()
        ax = plt.subplot(2, 2, 1)
        ax.plot(t, E, 'k')
        ax.set_ylabel('E')
        ax = plt.subplot(2, 2, 3)
        ax.plot(t, I, 'k')
        ax.set_ylabel('I')
        ax = plt.subplot(2, 2, (2, 4))
        ax.plot(E, I, 'k')
        ax.set_ylabel('I')
        ax.set_xlabel('E')

        x, y = np.meshgrid(np.linspace(0, 0.45, 30), np.linspace(-0.025, 0.33, 30))
        dEdt = (-x + (k_e - x)*sigmoid_function(c1*x - c2*y + P, a_e, theta_e))/tau_e
        dIdt = (-y + (k_i - y)*sigmoid_function(c3*x - c4*y + Q, a_i, theta_i))/tau_i
        ax = plt.subplot(2, 2, (2, 4))
        ax.quiver(x, y, dEdt, dIdt)
        plt.show()

        i = 0
        fig = plt.figure()
        for P in [1.065, 1.2, 1.8]:
            i += 1
            x = odeint(wilson_cowan, x0, t, args=(k_e, k_i, c1, c2, c3, c4,
                            tau_e, tau_i, a_e, a_i, theta_e, theta_i, P, Q))

            ax = plt.subplot(3, 1, i)
            ax.plot(t[:6000], x[:, 0][:6000], 'k')
            ax.set_title('Q = 0, P = {0:.3f}'.format(P))
        fig.suptitle('Excitatory population')
        plt.show()




    if args.connectivity == 0:
        logging.info('Unidirectional coupling Wilson-Cowan circuits')
        direction = 0
    else:
        logging.info('Bidirectional coupling Wilson-Cowan circuits')
        direction = 1

    # Some set of (alpha,beta). Stronger and weaker are just put as a way to distinguish them.
    if direction == 0:
        if args.free_parameters:
            alpha = float(input('Choose alpha: '))
            beta = float(input('Choose beta: '))
        else:
            strenght = input('Stronger or weaker E1-I2 coupling? [s/w]: ')
            if strenght == 's':
                alpha, beta = (5.3, -2.0)
            else:
                alpha, beta = (5.8, 0.1)
            logging.info(f'Chaotic regime --> alpha, beta = ({alpha}, {beta})')
    else:
        alpha = 1.3
        beta = 0
        logging.info(f'Chaoitc regime --> alpha = {alpha}')

    # initial conditions, integration step and time window
    x0 = [0.1, 0.1, 0.1, 0.1]
    step = 0.01
    t = np.arange(0, 300, step)

    if args.noisywc:
        if args.multiple_noisy_run:
            E1l, I1l, E2l, I2l = [], [], [], []
            run = 30
        else:
            run = 1
        for i in range(run):
            E1, I1, E2, I2, v = noisy_coupled_wc(alpha, beta, k_e, k_i, c1, c2, c3, c4, tau_e,
                                          tau_i, P1, P, Pp, direction, dt=step, time=t, x0=x0)
            E1l.append(E1)
            I1l.append(I1)
            E2l.append(E2)
            I2l.append(I2)
    else:
        x = odeint(coupled_wilson_cowan, x0, t, args=(alpha, beta,
            k_e, k_i, c1, c2, c3, c4, tau_e, tau_i, P1, P, Pp, direction))
        E1 = x[:, 0]
        I1 = x[:, 1]
        E2 = x[:, 2]
        I2 = x[:, 3]
    e1, e2, e3, e4 = Embed(E1), Embed(I1), Embed(E2), Embed(I2)

    if args.dynamics_figure:
        limit = int(len(t)/2)
        t = t[:limit]
        E1t, E2t, I1t, I2t = E1[:limit], E2[:limit], I1[:limit], I2[:limit]
        if args.noisywc:
            vt = v[:limit]
            fig = plt.figure()
            fig.suptitle('Time series')
            ax1 = plt.subplot(511)
            ax1.plot(t, E1t, 'k', lw=0.4)
            ax1.set_title('E1')
            ax2 = plt.subplot(512)
            ax2.plot(t, I1t, 'k', lw=0.4)
            ax2.set_title('I1')
            ax3 = plt.subplot(513)
            ax3.plot(t, E2t, 'k', lw=0.4)
            ax3.set_title('E2')
            ax4 = plt.subplot(514)
            ax4.plot(t, I2t, 'k', lw=0.4)
            ax4.set_title('I2')
            ax5 = plt.subplot(515)
            ax5.plot(t, vt, 'k', lw=0.4)
            ax5.hlines(v.mean(), 0, t[-1], 'g', '--')
            ax5.set_title('Noise')
            plt.show()
        else:
            fig = plt.figure()
            fig.tight_layout(pad=5.0)
            ax1 = plt.subplot(421)
            ax1.plot(t, E1t, 'k', lw=0.4)
            ax1.set_ylabel('E1')
            plt.setp(ax1.get_xticklabels(), fontsize=5)
            ax1.set_title('Time series')

            ax2 = plt.subplot(423)
            ax2.plot(t, I1t, 'k', lw=0.4)
            ax2.set_ylabel('I1')
            plt.setp(ax2.get_xticklabels(), fontsize=5)

            ax3 = plt.subplot(425)
            ax3.plot(t, E2t, 'k', lw=0.4)
            ax3.set_ylabel('E2')
            plt.setp(ax3.get_xticklabels(), fontsize=5)

            ax4 = plt.subplot(427)
            ax4.plot(t, I2t, 'k', lw=0.4)
            ax4.set_ylabel('I2')
            ax4.set_xlabel('Time')
            plt.setp(ax4.get_xticklabels(), fontsize=5)

            ax5 = plt.subplot(4,2,(6,8))
            ax5.plot(E2[10000:], I2[10000:], 'k', lw=0.4)
            ax5.set_xlabel('E2')
            ax5.set_ylabel('I2')
            ax5.grid()

            ax6 = plt.subplot(4, 2, (2, 4))
            ax6.plot(E1[:10000], I1[:10000], 'k', lw=0.4)
            ax6.set_xlabel('E1')
            ax6.set_ylabel('I1')
            ax6.set_title('2 dim phase space')
            ax6.grid()
            plt.show()

            ax = plt.subplot(projection='3d')
            ax.plot(E1[:10000], E2[:10000], I2[:10000], 'k', lw=0.4)
            ax.set_xlabel('E1')
            ax.set_ylabel('E2')
            ax.set_zlabel('I2')
            plt.title('3D phase space')
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
        plt.plot(t, E21, 'k--',
                 label=f'{x01}')
        plt.plot(t, E2, 'r', alpha=0.6,
                 label=f'{x0}')
        plt.grid()
        plt.legend()
        plt.show()

    if args.power_spectrum_:
        if args.noisywc:
            powerspectrum = [power_spectrum(E1, t, step), power_spectrum(I1, t, step),
                             power_spectrum(E2, t, step), power_spectrum(I2, t, step),
                             power_spectrum(v, t, step)]
            n = 5
        if args.singlewc:
            powerspectrum = [power_spectrum(E1[6000:], t[6000:], step), power_spectrum(I1[6000:], t, step)]
            ax = plt.subplot(2, 1, 1)
            ax.plot(powerspectrum[0][0], powerspectrum[0][1], 'k', lw=0.4)
            ax.set_title('E1')
            ax = plt.subplot(2, 1, 2)
            ax.plot(powerspectrum[1][0], powerspectrum[1][1], 'k', lw=0.4)
            ax.set_title('I1')
            plt.show()
        if direction == 1:
            n = 4
            powerspectrum = [power_spectrum(E1, t, step), power_spectrum(I1, t, step),
                             power_spectrum(E2, t, step), power_spectrum(I2, t, step)]

        ax = plt.subplot(n,1,1)
        ax.plot(powerspectrum[0][0], powerspectrum[0][1], 'k', lw=0.4)
        ax.set_title('E1')
        ax = plt.subplot(n,1,2)
        ax.plot(powerspectrum[1][0], powerspectrum[1][1], 'k', lw=0.4)
        ax.set_title('I1')
        ax = plt.subplot(n,1,3)
        ax.plot(powerspectrum[2][0], powerspectrum[2][1], 'k', lw=0.4)
        ax.set_title('E2')
        ax = plt.subplot(n,1,4)
        ax.plot(powerspectrum[3][0], powerspectrum[3][1], 'k', lw=0.4)
        ax.set_title('I2')
        if args.noisywc:
            ax = plt.subplot(n,1,5)
            ax.plot(powerspectrum[4][0], powerspectrum[4][1], 'k', lw=0.4)
            ax.set_title('Noise')
        ax.set_xlabel('Frequency')
        plt.show()

    # Mutual Information. It is calculated from skccm.Embed.mutual_information that relies of sklearn
    if args.mutual_information:
        fig, axs = plt.subplots(4)
        lag = 200
        axs[0].plot(e1.mutual_information(lag), 'k', lw=0.8)
        axs[0].set_title('E1')
        axs[1].plot(e2.mutual_information(lag), 'k', lw=0.8)
        axs[1].set_title('I1')
        axs[2].plot(e3.mutual_information(lag), 'k', lw=0.8)
        axs[2].set_title('E2')
        axs[3].plot(e4.mutual_information(lag), 'k', lw=0.8)
        axs[3].set_title('I2')
        axs[3].set_xlabel('Lags')
        fig.suptitle('Mutual Information')
        plt.show()


    if args.autocorr:
        # Maybe I can implement a test significance
        lag = 400
        auto_corrE1 = autocorrelation(E1, lag)
        auto_corrI1 = autocorrelation(I1, lag)
        auto_corrE2 = autocorrelation(E2, lag)
        auto_corrI2 = autocorrelation(I2, lag)
        plt.figure()
        plt.title('Autocorrelation')
        plt.plot(auto_corrE1, 'k', lw=0.8, label='E1')
        plt.plot(auto_corrI1, 'r', lw=0.8, label='I1')
        plt.plot(auto_corrE2, 'g', lw=0.8, label='E2')
        plt.plot(auto_corrI2, 'y', lw=0.8, label='I2')
        plt.grid()
        plt.legend()
        plt.show()

    if args.cross_correlation:
        plt.figure()
        crosscorr(E1, E2, 100, bootstrap_test=True)
        # crosscorr(E2, E1, 1000)
        # plt.legend(['E1->E2', 'E2->E1'])
        plt.show()


    # Set the embedding dimension and lag. Lag is choosen by looking at the first minimum in MI. It
    # represents the first time delay at which information is forgotten. Another way to select the lag
    # can be looking at the autocorrelation function, but it can give misleading results since the system
    # is not linear.
    if args.embedding:
        E2E1l, I2E1l, E2I1l, I2I1l, E1E2l = [], [], [], [], []
        lag = int(input('Select lag value (first minimum on MI): '))
        for E1, E2, I1, I2 in zip(E1l, E2l, I1l, I2l):
            embed_list = np.arange(1,10,1)
            bestE2E1, bestE1E2 = [], []
            bestI1I2, bestI2I1 = [], []
            bestE2I1, bestI1E2 = [], []
            bestE1I2, bestI2E1 = [], []
            bestE2I2, bestI2E2 = [], []
            for embed in tqdm(embed_list, desc='Embedding dimension list'):
                sc1ee, sc2ee, E1_emb, E2_emb, _ = prediction_skill(E1, E2, lag, embed)
                sc1ei1, sc2ei1, E1_emb, I2_emb, _ = prediction_skill(E1, I2, lag, embed)
                sc1ei2, sc2ei2, E2_emb, I1_emb, _ = prediction_skill(E2, I1, lag, embed)
                sc1ii, sc2ii, E2_emb, I1_emb, _ = prediction_skill(I1, I2, lag, embed)
                if direction == 0:
                    bestE2E1.append(sc1ee[-1]) # E1 -> E2
                    bestI2E1.append(sc1ei1[-1]) # E1 -> I2
                    bestE2I1.append(sc2ei2[-1]) # I1 -> E2
                    bestI2I1.append(sc1ii[-1]) # I1 -> I2
                    # Just to check that E2 does not drive E1
                    bestE1E2.append(sc2ee[-1])  # E2 -> E1

                    if args.multiple_noisy_run:
                        E2E1l.append(bestE2E1)
                        I2E1l.append(bestI2E1)
                        E2I1l.append(bestE2I1)
                        I2I1l.append(bestI2I1)
                        E1E2l.append(bestE1E2)
                else:
                    bestE1E2.append(sc2ee[-1]) # E2 -> E1
                    bestE1I2.append(sc2ei1[-1]) # I2 -> E1
                    bestI1E2.append(sc1ei2[-1]) # E2 -> I1
                    bestI1I2.append(sc2ii[-1]) # I2 -> I1
                    bestE2E1.append(sc1ee[-1])  # E1 -> E2
                    bestI2E1.append(sc1ei1[-1])  # E1 -> I2
                    bestE2I1.append(sc2ei2[-1])  # I1 -> E2
                    bestI2I1.append(sc1ii[-1])  # I1 -> I2
                

        plt.figure()
        plt.title('Prediction skills as function of embedding dimension')
        if direction == 0:
            if args.multiple_noisy_run:
                plt.plot(embed_list, np.mean(E2E1l, axis=0),
                        'red', lw=0.2, label='E1 => E2')
                plt.plot(embed_list, np.mean(E2E1l, axis=0) +
                        np.std(E2E1l, axis=0), 'red', lw=0.2)
                plt.plot(embed_list, np.mean(E2E1l, axis=0) -
                        np.std(E2E1l, axis=0), 'red', lw=0.2)
                plt.fill_between(np.arange(1, len(embed_list)+1), np.mean(
                    E2E1l, axis=0)+np.std(E2E1l, axis=0), np.mean(E2E1l, axis=0)-np.std(E2E1l, axis=0), color='red', alpha=0.6)

                plt.plot(embed_list, np.mean(E1E2l, axis=0),
                        'purple', lw=0.2, label='E2 => E1')
                plt.plot(embed_list, np.mean(E1E2l, axis=0) +
                        np.std(E1E2l, axis=0), 'purple', lw=0.2)
                plt.plot(embed_list, np.mean(E1E2l, axis=0) -
                        np.std(E1E2l, axis=0), 'purple', lw=0.2)
                plt.fill_between(np.arange(1, len(embed_list)+1), np.mean(
                    E1E2l, axis=0)+np.std(E1E2l, axis=0), np.mean(E1E2l, axis=0)-np.std(E1E2l, axis=0), color='purple', alpha=0.6)

                plt.plot(embed_list, np.mean(E2I1l, axis=0),
                        'aqua', lw=0.2, label='I1 => E2')
                plt.plot(embed_list, np.mean(E2I1l, axis=0) +
                        np.std(E2I1l, axis=0), 'aqua', lw=0.2)
                plt.plot(embed_list, np.mean(E2I1l, axis=0) -
                        np.std(E2I1l, axis=0), 'aqua', lw=0.2)
                plt.fill_between(np.arange(1, len(embed_list)+1), np.mean(
                    E2I1l, axis=0)+np.std(E2I1l, axis=0), np.mean(E2I1l, axis=0)-np.std(E2I1l, axis=0), color='aqua', alpha=0.6)

            else:
                plt.plot(embed_list, bestE2E1, label="E1 => E2")
                plt.plot(embed_list, bestI2I1, label="I1 => I2")
                plt.plot(embed_list, bestE2I1, label="I1 => E2")
                plt.plot(embed_list, bestI2E1, label="E1 => I2")
                plt.plot(embed_list, bestE1E2, label="E2 => E1")
        else:
            plt.plot(embed_list, bestE2E1, label="E1 => E2")
            plt.plot(embed_list, bestI2I1, label="I1 => I2")
            plt.plot(embed_list, bestE2I1, label="I1 => E2")
            plt.plot(embed_list, bestI2E1, label="E1 => I2")
            plt.plot(embed_list, bestE1E2, label="E2 => E1")
            plt.plot(embed_list, bestI1I2, label="I2 => I1")
            plt.plot(embed_list, bestI1E2, label="E2 => I1")
            plt.plot(embed_list, bestE1I2, label="I2 => E1")
            
        plt.ylabel('Coefficient of determination')
        plt.xlabel('Dimension')
        plt.grid()
        plt.legend()
        plt.show()

        embed = int(input('Choose dimension of the embedding space: '))
        sc1ee, sc2ee, E1emb, E2emb, lib_lens_ee = prediction_skill(E1, E2, lag, embed)
        sc1ii, sc2ii, I1emb, I2emb, lib_lens_ii = prediction_skill(I1, I2, lag, embed)

        fig = plt.figure()
        plt.title(f'Embedded attractors. Dimension equal to {embed}')
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter(E1emb[:, 0][2000:], E1emb[:, 1][2000:], E1emb[:, 2][2000:], 'k', s=0.1)
        ax.set_xlabel('E1(t)')
        ax.set_ylabel(f'E1(t+{lag})')
        ax.set_zlabel(f'E1(t+2*{lag})')

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.scatter(E2emb[:, 0][2000:], E2emb[:, 1][2000:], E2emb[:, 2][2000:], 'k', s=0.1)
        ax.set_xlabel('E2(t)')
        ax.set_ylabel(f'E2(t+{lag})')
        ax.set_zlabel(f'E2(t+2*{lag})')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(I1emb[:, 0][2000:], I1emb[:, 1][2000:], I1emb[:, 2][2000:], 'k', s=0.1) 
        ax.set_xlabel('I1(t)')
        ax.set_ylabel(f'I1(t+{lag})')
        ax.set_zlabel(f'I1(t+2*{lag})')

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(I2emb[:, 0][2000:], I2emb[:, 1][2000:], I2emb[:, 2][2000:], 'k', s=0.1)
        ax.set_xlabel('I2(t)')
        ax.set_ylabel(f'I2(t+{lag})')
        ax.set_zlabel(f'I2(t+2*{lag})')


        plt.figure()
        plt.title('Prediction skill as function of library lenght')
        plt.plot(lib_lens_ee, sc1ee, 'g', label='E1 => E2')
        plt.plot(lib_lens_ii, sc2ee, 'r', label='E2 => E1')
        plt.xlabel('Library lenght')
        plt.grid()
        plt.legend()
        plt.show()
