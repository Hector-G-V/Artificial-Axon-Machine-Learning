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


def p_inf(n_nodes):
    """
    Calculates the channel states for t<0 and prepends the resting potential to the returned array.
    The resting potential is equal to the CLVC value.
    :param n_nodes: number of nodes in the network.
    :return: V_r and channel states for t<0.
    """
    V_r = -150 * 10 ** -3  # Resting potential for t<0.
    d, p_o = np.array([]), np.array(V_r)

    for j in range(7):
        d = np.append(d, np.linalg.det(np.delete(np.delete(Q(V_r), j, 0), j, 1)))

    for j in range(7):
        p_o = np.append(p_o, d[j] / np.sum(d))

    for j in range(n_nodes):
        p_o = np.append(p_o, p_o, axis=None)

    return p_o


def start_pulse(t):
    """
    Current Clamp (CC) pulse to induce an Axon to fire.
    This function is intended to start only the very first node in the entire network.
    :param t: Time argument.
    :return: Pulse at start time t_s with time width t_w.
    """

    t_s = 0.25  # Pulse start time.
    t_w = 0.65  # Pulse width.
    I_c = 60 * 10 ** -12  # CC value.

    return I_c*np.heaviside(t - t_s, 0)*np.heaviside(t_s + t_w - t, 0)


def node(y, t):
    """
    System of membrane potential and channel state differential equations.
    Two Axons connected by synapses comprise one node.
    Axon with variable subscript 'u' (for 'up') faces the rest of the network.
    Axon with variable subscript 'd' (for 'down') functions as the second channel species in a biological neuron.
    :param y: All time-dependent variables.
    :param t: Time steps.
    :return: System of equations.
    """
    # Axon Constants.
    V_c, V_N = -150 * 10 ** -3, 60 * 10 ** -3  # CLVC voltage and Nernst potential.
    R_c, R_s = 2 * 10 ** 9, 5 * 10 ** 8  # CLVC and Synapse resistor values.
    C = 150 * 10 ** -12  # Membrane capacitance.
    N_o = 10 ** 3  # Number of channels in one Artificial Axon.
    X = (6 * 10 ** 9) ** -1  # Single KvAP channel conductance.

    # Time-dependent variables.
    V_u, o_u, i_u, c4_u, c3_u, c2_u, c1_u, c0_u, V_d, o_d, i_d, c4_d, c3_d, c2_d, c1_d, c0_d = y

    # Equations.
    dydt = [
        (1/C)*(N_o*X*o_u*(V_N - V_u) + (1/R_c)*(V_c - V_u) - (1/R_s)*V_d*np.heaviside(V_d, 0) + start_pulse(t)),
        c4_u*k_co(V_u) - o_u*k_oc(V_u),
        c4_u*k_ci(V_u) - i_u*k_ic(V_u),
        i_u*k_ic(V_u) + o_u*k_oc(V_u) + c3_u*a(V_u) - c4_u*(k_co(V_u) + k_ci(V_u) + 4*B(V_u)),
        c4_u*4*B(V_u) + c2_u*2*a(V_u) - c3_u*(1*a(V_u) + 3*B(V_u)),
        c3_u*3*B(V_u) + c1_u*3*a(V_u) - c2_u*(2*a(V_u) + 2*B(V_u)),
        c2_u*2*B(V_u) + c0_u*4*a(V_u) - c1_u*(3*a(V_u) + 1*B(V_u)),
        c1_u*1*B(V_u) - c0_u*4*a(V_u),

        (1/C)*(N_o*X*o_d*(V_N - V_d) + (1/R_c)*(V_c - V_d) + (1/R_s)*V_u*np.heaviside(V_u, 0)),
        c4_d * k_co(V_d) - o_d * k_oc(V_d),
        c4_d * k_ci(V_d) - i_d * k_ic(V_d),
        i_d * k_ic(V_d) + o_d * k_oc(V_d) + c3_d * a(V_d) - c4_d * (k_co(V_d) + k_ci(V_d) + 4 * B(V_d)),
        c4_d * 4 * B(V_d) + c2_d * 2 * a(V_d) - c3_d * (1 * a(V_d) + 3 * B(V_d)),
        c3_d * 3 * B(V_d) + c1_d * 3 * a(V_d) - c2_d * (2 * a(V_d) + 2 * B(V_d)),
        c2_d * 2 * B(V_d) + c0_d * 4 * a(V_d) - c1_d * (3 * a(V_d) + 1 * B(V_d)),
        c1_d * 1 * B(V_d) - c0_d * 4 * a(V_d)
    ]

    return dydt


def plot(solution, t):
    """
    Solution Plot.
    :param solution:
    :param t: Time steps.
    :return: Membrane potential V(t) vs t on one plot, Open and Inactive states vs t on another.
    """
    plt.subplot(2, 2, 1)
    plt.plot(t, sol[:, 0], label='V(t)')
    plt.legend(loc='best')
    plt.ylabel('V')
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(t, sol[:, 1], label='o(t)')
    plt.plot(t, sol[:, 2], label='i(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(t, sol[:, 8], label='V(t)')
    plt.legend(loc='best')
    plt.ylabel('V')
    plt.grid()  # Could absolutely make this smaller. But leave as-is, for now.
    plt.subplot(2, 2, 4)
    plt.plot(t, sol[:, 9], label='o(t)')
    plt.plot(t, sol[:, 10], label='i(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # Args.
    t_i, t_f = 0, 3  # Simulation start and end times.

    # Solve.
    y0 = p_inf(1)  # Initial Values. All units are SI.
    t_steps = np.linspace(t_i, t_f, 10001)  # Time steps.
    sol = odeint(node, y0, t_steps)  # Solves the system of equations.

    # Plot.
    plot(sol, t_steps)

