"""
Runs a network that uses two Artificial Axons as one node.
Artificial Axon simulation uses the Hodgkin-Huxley model and ion channel KvAP.

Two Axons connected by synapses comprise one node.
One node Axon faces the rest of the network.
The second node Axon functions as the second channel species in a biological neuron.
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
    p_net = np.ones(n_nodes * 2) * V_r  # Array holds initial conditions for entire network.
    d, p_o = np.array([]), np.array([])  # For calculation of channel states in one Axon.

    for j in range(7):
        d = np.append(d, np.linalg.det(np.delete(np.delete(Q(V_r), j, 0), j, 1)))

    for j in range(7):  # p_o: channel states at t<0 for one Axon.
        p_o = np.append(p_o, d[j] / np.sum(d))

    for j in range(n_nodes):  # Initial conditions for all Axons in the network.
        p_net = np.append(p_net, p_o, axis=None)  # Channel states for 1st Axon in a node.
        p_net = np.append(p_net, p_o, axis=None)  # Channel states for 2nd Axon in a node.

    return p_net


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


def node_counts(matrices):
    """
    Returns a count of nodes in each matirx.
    The first matrix is for the initialization nodes. Really only one initialization node is necessary.
    The second matrix is for the input layer.
    Matrices 2 through second-from-last are fo the hidden layers.
    The final matrix is for the output layer.
    :param matrices: A list of all weights matrices.
    :return: List of node count for each matrix.
    """

    nodes_array = []  # Holds the number of nodes in each layer.
    n_in, n_int = np.shape(syn[0])  # Number of input, initialization nodes = Number of rows, columns in matrix 0.
    nodes_array.append(n_int)
    nodes_array.append(n_in)

    for j in range(1, len(syn)-1):
        n_hidden = np.shape(syn[j])[0]  # Number of hidden layer nodes = Number of rows in matrix j.
        nodes_array.append(n_hidden)

    n_out = np.shape(syn[-1])[0]  # Number of output nodes = Number of rows in final matrix.
    nodes_array.append(n_out)

    return nodes_array


def network(y, t):
    """
    System of membrane potential and channel state differential equations.
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

    n_nodes = sum(nodes)  # Number of nodes.
    n_axons = n_nodes*2  # Number of Axons. Also equal to int(len(y)/8).

    Syn = np.array([])

    for h in range(len(nodes)-1):
        a_0 = np.array([])  # a_0 layer vector.
        for j in range(nodes[h]):
            k = 2*(sum(nodes[:h]) + j)  # Index for the membrane equation in question.
            a_0 = np.append(a_0, y[k]*np.heaviside(y[k], 0))  # a_0 element = V*Step(V)
        a_1 = np.matmul(syn[h]/R_s, a_0)  # a_1 layer vector.
        Syn = np.append(Syn, a_1)

    for j in range(len(Syn)):  # Caps the current injected into the post-synaptic neuron.
        if Syn[j] > 0.05871*0.85/R_s:
            Syn[j] = 0.05871*0.71/R_s

    Syn = np.insert(Syn, 0, np.zeros(nodes[0]))  # The first nodes in the y array are the initialization nodes.
    for j in range(nodes[0]):  # Initialization nodes initialize the input layer.
        Syn[j] = start_pulse(t)

    dydt = np.array([])

    for j in range(n_nodes):
        k = 2*j
        dydt = np.append(dydt,
                         (1/C)*(N_o*X*y[n_axons+7*k]*(V_N - y[k]) + (1/R_c)*(V_c - y[k])
                                - (1/R_s)*y[k+1]*np.heaviside(y[k+1], 0) + Syn[j]))
        dydt = np.append(dydt,
                         (1/C)*(N_o*X*y[n_axons+7*(k+1)]*(V_N - y[k+1]) + (1/R_c)*(V_c - y[k+1])
                                + (1/R_s)*y[k]*np.heaviside(y[k], 0)))

    for j in range(n_axons):
        dydt = np.append(dydt, np.matmul(Q(y[j]).T, y[n_axons+7*j:n_axons+7*(j+1)]), axis=None)

    return dydt


def plot(solution, t):
    """
    Solution Plot.
    :param solution: Solved system of equations.
    :param t: Time steps.
    :return: Membrane potential V(t) vs t on one plot, Open and Inactive states vs t on another.
    """

    layer = ['Initialization Layer', 'Input Layer']
    for j in range(len(nodes) - 3):
        k = j+1
        layer.append("Hidden Layer %d" % k)
    layer.append('Output Layer')

    for h in range(len(layer)):
        plt.figure(layer[h])
        for j in range(nodes[h]):
            plt.subplot(nodes[h], 1, j+1)
            plt.plot(t, sol[:, 2*(sum(nodes[:h]) + j)], label='V0_%d' % j)
            plt.legend(loc='best')
            plt.ylabel('V')
            plt.grid()
        plt.xlabel('t')

    plt.show()


if __name__ == '__main__':

    # Synapse initialization.
    syn = [np.array([[0.8]]), np.array([[0]])]
    nodes = node_counts(syn)
    t_i, t_f = 0, 1 + 0.5*(len(nodes) + 1)  # Simulation start and end times.

    # Solve.
    y0 = p_inf(sum(nodes))  # Initial Values. All units are SI.
    t_steps = np.linspace(t_i, t_f, 10001)  # Time steps.
    sol = odeint(network, y0, t_steps)  # Solves the system of equations.

    # Plot.
    plot(sol, t_steps)
