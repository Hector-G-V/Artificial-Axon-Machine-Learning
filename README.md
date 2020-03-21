Produce a Feedforward neural network with an Artificial Axon.

# Background

## The Artificial Axon
The Artificial Axon is a research apparatus that uses ion channel proteins in a mechanical and electronic hardware system to produce an action potential outside of the biological neuron. Ion channels are suspended on a lipid membrane in the system as shown in the figure below.

<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/KvSetup.png" width="480" height="270">
</p>

## The Model
The membrane potential is governed by the following equation:
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/Membrane Equation.png" width="50%" height="50%">
</p>

The ion channel model is a system of ordinary differential equations that represents change in the ion channel’s inner states across time. The equations are:
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/Channel Matrix.png" width="125%" height="125%">
</p>

The equations above complete the model for the membrane potential and channel states in the Artificial Axon.

The figure below shows one measurement of an action potential in the Artificial Axon. The blue trace is the membrane potential, and the yellow trace is a fit using the system of equations above.
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/KvAPFit.png" width="50%" height="50%">
</p>

## One Node
Two Axons are connected with electronics to produce one “node,” the basic computational unit in a neuron-like network. One Axon (Axon 1) in the node faces the network – receives input from other nodes – and the second Axon (Axon 2) functions as the second channel species in a biological neuron, pulling the first Axon’s membrane potential down to rest after an action potential, so that the channels may recover from inactivation to fire again.

The diagram below shows an example of a node’s response to an external input pulse. Axon 1 provides excitatory input to Axon 2, and Axon 2 provides inhibitory input to Axon 1.
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/One Node.png" width="35%" height="35%">
</p>

The electronic connection between the Axons provides input as follows:
<p align="center">Input to Axon 2 &rarr; +V<sub>1</sub> &Theta;(V<sub>1</sub>)</p>
<p align="center">Input to Axon 1 &rarr; -V<sub>2</sub> &Theta;(V<sub>2</sub>)</p>
<p> where &Theta; is the unit step function. The post-synaptic Axon will only fire when the pre-synaptic membrane potential <i>V<sub>pre</sub> > 0</i>. </p>

This node in a network can be trained with the backpropagation algorithm to perform deep learning tasks. Below I describe how this naturally emerges from an Artificial Axon network.

# Artificial Axon Network
This module simulates a network of Artificial Axon nodes. The module input is a list of matrices for each layer in the network, and the module output is all membrane and channel values across time.

<p>
The first layer is an initialization layer, meaning it is intended to initialize the input layer. Every node in the input layer has a special input <i>start_pulse(t)</i>. This is a square pulse, <i>60 picoAmps</i> tall and <i>650 milliseconds</i> wide, that begins at <i>t = 250 milliseconds</i>. This initialization layer receives no input and has no other connections.
</p>
The second layer is the input layer. Every layer that follows is a hidden layer, except for the final output layer.
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/Artificial Axon NN Graph.png" width="100%" height="100%">
</p>

## Activation
The input to a node in the hidden or output layers has the following form:
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/Syn.png" width="20%" height="20%">
</p>
<p> 
where <i>Syn</i> is short for "synapse," <i>R<sub>s</sub></i> is a constant that represents a resistor value in the real Artificial Axon system, <i>&theta;<sub>j</sub></i> is an individual weight, and &Theta; is the unit step function.
</p>

<p>
Note that the synaptic input depends on the shape of the pre-synaptic potential <i>V(t)</i>. If all pre-synaptic nodes fire at different times, it would be impossible to find a simple prediction scheme for the post-synaptic firing threshold. Two implementations resolve this issue.

First: the initialization layer is designed to give all input nodes identical stimulus. If all the input nodes receive the same initialization stimulus, then the resulting <i>V(t)</i> shape of all firing input nodes becomes identical as well. Second, a stimulus maximum is imposed on the stimulus sent to a post-synaptic node. This means that if the stimulus is larger than some set maximum value of our choosing, then the stimulus becomes flat. The reason this is important is as follows. Unlike machine learning Artificial Neural Networks, spike timing is important in an Artificial Axon network. Without a stimulus cap, all nodes in a layer would fire at different times. These time differences become larger as the number of layers increases, which could make the output information difficult or impossible to interpret. However, an appropriately chosen cap prevents timing shifts, and the cap options are simple to implement.

Given these two implementations, all membrane potentials are nearly identical at every point in time during an action potential. For the <i>m</i> pre-synaptic nodes that fire, the synaptic stimulus becomes:
</p>
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/Syn Fire.png" width="20%" height="20%">
</p>
<p>
where <i>V(t)</i> is the membrane potential value of the pre-synaptic node at any time <i>t</i>. Given this form, it becomes a simple task to find a &Sigma;<i>&theta;<sub>k</sub></i> value that would induce an action potential in a post-synaptic node. For the ion channel parameters chosen in this module, a node fires when it receives stimulus &Sigma;<i>&theta;<sub>k</sub>=0.8</i>.
</p>
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/Syn Sum Theta.png" width="100%" height="100%">
</p>

Given the ion channel parameters in this module, the chosen cap is as follows:
```
for j in range(len(Syn)):  # Caps the current injected into the post-synaptic neuron.
  if Syn[j] > 0.05871*0.85/R_s:
  Syn[j] = 0.05871*0.71/R_s
```

A note on the physical Artificial Axon system: the current-capping electronic circuit would not be difficult to implement. A possible solution is to connect all inputs to an op-amp that saturates at some specified picoAmp value. The op-amp output is then sent to the synapse circuit for the post-synaptic node.

# Feedforward Neural Network in TensorFlow
A single Artificial Axon node behaves like a node in a Feedforward deep learning network with a sigmoid activation step. For this reason, a network of AA nodes can be trained using the backpropagation algorithm. Furthermore, any task that can be solved with a deep learning network can also be solved with an AA network, and this module demonstrates how. In this module, an FFNN is trained to detect handwritten digits, a simple and well-known neural network problem and solution. Two implementations make this an Artificial Axon network. First is the input: each pixel value is set to either 0 or 1. A neuron either fires or it does not – there is no in-between. Second: the activation is a sigmoid centered at value 0.8.

# RNN with the Artificial Axon
One method for implementing an RNN with an Aritificial Axon network is to use a series of connected Artificial Axon nodes as a clock, fixing the interval at which input can be received. The diagram below shows how this can be done.
</p>
<p align="center">
  <img src="https://github.com/Hector-G-V/Images/blob/master/OscRNN.png" width="100%" height="100%">
</p>
