"""
Artificial Axon simulation with the Hodgkin-Huxley model and ion channel KvAP.
"""

import numpy as np
from numpy import exp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Rate Constant multipliers to fit a channel from an Artificial Axon experiment.
a_c, b_c = 9, 1.8


def a(v):
    """
    1 of 6 Rate Constants.
    This Constant is for the transition from Closed state C_beta to C_alpha.
    :param v: Membrane potential.
    :return: Rate Constant value.
    """
    return a_c*0.06*54.4*exp(26*v)


def B(v):
    """
    2 of 6 Rate Constants.
    This Constant is for the transition from Closed state C_alpha to C_beta.
    :param v: Membrane potential.
    :return: Rate Constant value.
    """
    return a_c*(1/0.06)*(11*10**-3)*exp(-67*v)


def k_ci(v):
    """
    3 of 6 Rate Constants.
    This Constant is for the transition from Closed state C_4 to the Inactive state.
    :param v: Membrane potential.
    :return: Rate Constant value.
    """
    return b_c*17.7*exp(v)


def k_ic(v):
    """
    4 of 6 Rate Constants.
    This Constant is for the transition from the Inacive state to Closed state C_4.
    :param v: Membrane potential.
    :return: Rate Constant value.
    """

    return b_c*(8*10**-3)*exp(-30*v)


def k_co(v):
    """
    5 of 6 Rate Constants.
    This Constant is for the transition from Closed state C_4 to the Open state.
    :param v: Membrane potential.
    :return: Rate Constant value.
    """
    return a_c*12.4*exp(-v)


def k_oc(v):
    """
    6 of 6 Rate Constants.
    This Constant is for the transition from the Open state to Closed state C_4.
    :param v: Membrane potential.
    :return: Rate Constant value.
    """
    return a_c*9.8*exp(-12*v)


def Q(v):
    """
    The Q-Matrix. For finding the channel states for t<0.
    :param v: Membrane potential
    :return: The Q-Matrix.
    """
    return np.array(
        [
            [-k_oc(v),    0   ,          k_oc(v)           ,       0       ,         0       ,         0     ,    0   ],
            [   0    ,-k_ic(v),          k_ic(v)           ,       0       ,         0       ,         0     ,    0   ],
            [ k_co(v), k_ci(v), -k_ci(v) - k_co(v) - 4*B(v),     4*B(v)    ,         0       ,         0     ,    0   ],
            [   0    ,    0   ,           a(v)             , -a(v) - 3*B(v),       3*B(v)    ,         0     ,    0   ],
            [   0    ,    0   ,            0               ,     2*a(v)    , -2*a(v) - 2*B(v),       2*B(v)  ,    0   ],
            [   0    ,    0   ,            0               ,       0       ,       3*a(v)    , -3*a(v) - B(v),   B(v) ],
            [   0    ,    0   ,            0               ,       0       ,         0       ,       4*a(v)  , -4*a(v)],
        ]
    )


def p_inf(v):
    """
    Calculates the channel states for t<0.
    :param v: Membrane potential.
    :return: Channel states for t<0.
    """
    d, p_o = np.array([]), np.array([])

    for j in range(7):
        d = np.append(d, np.linalg.det(np.delete(np.delete(Q(v), j, 0), j, 1)))

    for j in range(7):
        p_o = np.append(p_o, d[j] / np.sum(d))

    return p_o


def axon(y, t):
    """
    System of membrane potential and channel state differential equations.
    :param y: All time-dependent variables.
    :param t: Time steps.
    :return: System of equations.
    """
    # Axon Constants.
    V_c, V_N = -51 * 10 ** -3, 60 * 10 ** -3  # CLVC voltage and Nernst potential.
    R_c, R_s = 2 * 10 ** 9, 5 * 10 ** 8  # CLVC and Synapse resistor values.
    C = 150 * 10 ** -12  # Membrane capacitance.
    N_o = 10 ** 3  # Number of channels in one Artificial Axon.
    X = (6 * 10 ** 9) ** -1  # Single KvAP channel conductance.

    # Time-dependent variables.
    V, o, i, c4, c3, c2, c1, c0 = y

    # Equations.
    dydt = [
        (1/C)*(N_o*X*o*(V_N - V) + (1/R_c)*(V_c - V)),
        c4*k_co(V) - o*k_oc(V),
        c4*k_ci(V) - i*k_ic(V),
        i*k_ic(V) + o*k_oc(V) + c3*a(V) - c4*(k_co(V) + k_ci(V) + 4*B(V)),
        c4*4*B(V) + c2*2*a(V) - c3*(1*a(V) + 3*B(V)),
        c3*3*B(V) + c1*3*a(V) - c2*(2*a(V) + 2*B(V)),
        c2*2*B(V) + c0*4*a(V) - c1*(3*a(V) + 1*B(V)),
        c1*1*B(V) - c0*4*a(V)
    ]

    return dydt


def plot(solution, t):
    """
    Solution Plot.
    :param solution:
    :param t: Time steps.
    :return: Membrane potential V(t) vs t on one plot, Open and Inactive states vs t on another.
    """
    plt.subplot(2, 1, 1)
    plt.plot(t, sol[:, 0], label='V(t)')
    plt.legend(loc='best')
    plt.ylabel('V')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t, sol[:, 1], label='o(t)')
    plt.plot(t, sol[:, 2], label='i(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # Args.
    V_r = -150 * 10 ** -3  # Resting potential for t<0.
    t_i, t_f = 0, 3  # Simulation start and end times.

    # Solve.
    y0 = np.concatenate((V_r, p_inf(V_r)), axis=None)  # Initial Values. All units are SI.
    t_steps = np.linspace(t_i, t_f, 10001)  # Time steps.
    sol = odeint(axon, y0, t_steps)  # Solves the system of equations.

    # Plot.
    plot(sol, t_steps)
