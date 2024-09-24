import numpy as np
import scipy as sc
import time
from qiskit_entropy.utils import *
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from torch.autograd import Variable
import json

# パラメータを取得
config = load_config('/home/mitsukism/qiskit_entropy/vn_ent_qubits10_training_config.json')

qubits = config["qubits"]
num_wires = config["num_wires"]
num_layers = config["num_layers"]
N = config["N"]
seed = config["seed"]
num_shots = config["num_shots"]
num_of_epochs = config["num_of_epochs"]
learning_rate = config["learning_rate"]
num_of_samples = config["num_of_samples"]
dimension = config["dimension"]
hidden_layer = config["hidden_layer"]

device = qml.device("default.qubit", wires=num_wires, shots=num_shots)
@qml.qnode(device)
def measure_rho(param, circuit_structure, qubits, rotations=[qml.RX, qml.RY, qml.RZ]):
    obj_wires = range(qubits)   

    qml.Hadamard(wires=0)

    for gate_info in circuit_structure:
        gate = gate_info["gate"]
        wires = gate_info["wires"]
        if gate == "CNOT":
            qml.CNOT(wires=wires)
        elif gate == "PauliX":
            qml.PauliX(wires=wires[0])
        elif gate == "PauliY":
            qml.PauliY(wires=wires[0])
        elif gate == "PauliZ":
            qml.PauliZ(wires=wires[0])

    qml.RandomLayers(param, wires=obj_wires, rotations=rotations)

    result = [qml.sample(qml.PauliZ(i)) for i in range(len(obj_wires))]
    return result

class neural_function(nn.Module):
    def __init__(self,dimension,hidden_layer):
        super(neural_function, self).__init__()

        self.dimension = dimension
        self.hidden_layer = hidden_layer
        self.lin1 = nn.Linear(self.dimension, self.hidden_layer)
        self.lin_end = nn.Linear(self.hidden_layer, 1)

    def forward(self, input):
        y = torch.sigmoid(self.lin1(input.float()))
        y = self.lin_end(y)

        return y
    
#@title Optimization using Gradient Descent (with neural network)
neural_fn = neural_function(dimension, hidden_layer)
param_init = np.random.random(qml.RandomLayers.shape(n_layers=num_layers, n_rotations=3))
circuit_structure = load_circuit_structure_from_json('/home/mitsukism/qiskit_entropy/circuit_structure.json')
print(circuit_structure)

cost_func_store = []
# start the training
for epoch in range(1, num_of_epochs):
    
  # evaluate the gradient with respect to the quantum circuit parameters
    gradients = np.zeros_like((param_init))
    
    for i in range(len(gradients)):
        for j in range(len(gradients[0])):

      # copy the parameters
            shifted = param_init.copy()

      # right shift the parameters
            shifted[i, j] += np.pi/2

      # forward evaluation
            forward_sum = 0
            result = measure_rho(shifted, circuit_structure, qubits)
            for sample in range(num_of_samples):
                sample_result_array = np.array([result[q][sample] for q in range(dimension)])
                nn_result = neural_fn(torch.from_numpy(sample_result_array))
                forward_sum += nn_result[0].detach().numpy()

      # normalize the forward sum
            forward_sum = forward_sum/num_of_samples

      # left shift the parameters
            shifted[i, j] -= np.pi

      # backward evaluation
            backward_sum = 0
            result = measure_rho(shifted, circuit_structure, qubits)
            for sample in range(num_of_samples):
                sample_result_array = np.array([result[q][sample] for q in range(dimension)])
                nn_result = neural_fn(torch.from_numpy(sample_result_array))
                backward_sum += nn_result[0].detach().numpy()

      # normalize the backward sum
            backward_sum = backward_sum/num_of_samples
      #print(backward_sum)

      # parameter-shift rule
            gradients[i, j] = - 0.5 * (forward_sum - backward_sum)
    np.save(f"/home/mitsukism/qiskit_entropy/vn_net_gradients/qubits10/gradients_epoch{epoch}_seed{seed}.npy", gradients)

  # first copy the quantum circuit parameters before updating it
    prev_param_init = param_init.copy()

  # update the quantum circuit parameters
    param_init -= learning_rate*gradients

  # evaluate the gradient with respect to the NN parameters
    optimizer = optim.SGD(neural_fn.parameters(), lr=learning_rate)

  # evaluate the first term
    loss = 0
    result = measure_rho(prev_param_init, circuit_structure, qubits)
    for sample in range(num_of_samples):
        # optimizer.zero_grad()
        sample_result_array = np.array([result[q][sample] for q in range(dimension)])
        random_result_array = np.random.choice([-1, 1], size=dimension)
        sample_nn_result = neural_fn(torch.from_numpy(sample_result_array))
        random_nn_result = neural_fn(torch.from_numpy(random_result_array))
        loss_term = (torch.exp(random_nn_result[0]) - sample_nn_result[0]).to("cpu")
        loss += loss_term / num_of_samples
    loss.backward()
    torch.nn.utils.clip_grad_norm_(neural_fn.parameters(), max_norm=1.0)
    optimizer.step()

  # evaluate the cost function at these parameters
    first_term = 0
    result = measure_rho(param_init, circuit_structure, qubits)
    for sample in range(num_of_samples):
        sample_result_array = np.array([result[q][sample] for q in range(dimension)])
        nn_result = neural_fn(torch.from_numpy(sample_result_array))
        first_term += nn_result[0].detach().numpy()

  # normalize the cost sum
    first_term = first_term/num_of_samples

  # # Second term evaluation
    second_term = 0
    for sample in range(num_of_samples):
        result = np.random.choice([-1, 1], size=dimension)
        nn_result = neural_fn(torch.from_numpy(result.flatten()))
        second_term += np.exp(nn_result[0].detach().numpy())

  # normalize the second term sum
    second_term = second_term/num_of_samples

    # add the cost function to the store
    cost_func_store.append(np.log(N) - first_term + second_term - 1)
  # save the cost function
    np.save(f"/home/mitsukism/qiskit_entropy/vn_net_cost/qubits10/cost_epoch{epoch}_seed{seed}.npy", cost_func_store)