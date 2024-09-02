import numpy as np
import cvxpy as cp
import scipy as sc
import time
import argparse
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import json

parser = argparse.ArgumentParser(description='Variational Quantum Entropy Estimation')

parser.add_argument('--qubits', type=int, default=2, help='Number of qubits')
parser.add_argument('--num_wires', type=int, default=4, help='Number of wires')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--num_shots', type=int, default=100, help='Number of shots')
parser.add_argument('--N', type=int, default=4, help='Number of samples')
parser.add_argument('--num_of_epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
parser.add_argument('--num_of_samples', type=int, default=100, help='Number of samples')
parser.add_argument('--dimension', type=int, default=2, help='Dimension')
parser.add_argument('--hidden_layer', type=int, default=2, help='Hidden layer')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--save_circuit_structure', type=bool, default=False, help='Save circuit structure to JSON file')
parser.add_argument('--load_circuit_structure_path', type=str, default=None, help='Load circuit structure from JSON file')

args = parser.parse_args()

def claculate_entropy(rho):
    sigma = np.eye(args.N) / args.N
    np.seterr(divide='ignore', invalid='ignore')
    H_rho = np.real(np.trace(rho @ (sc.linalg.logm(rho) - sc.linalg.logm(sigma))))
    np.seterr(divide = 'warn', invalid='warn')
    return H_rho

def generate_random_circuit_structure(qubits, ratio_imprim=0.8, pauli_gates=['PauliX', 'PauliY', 'PauliZ'], seed=None):
    if seed:
        np.random.seed(seed)

    obj_wires = range(qubits)   
    num_qubits = len(obj_wires) * 2
    rng = np.random.default_rng(seed)

    num_gates = len(obj_wires) * len(pauli_gates)

    random_values = rng.random(3 * num_gates)
    random_choices = random_values[:num_gates] 
    gate_indices = (random_values[num_gates:2 * num_gates] * len(pauli_gates)).astype(int)
    wire_indices = (random_values[2 * num_gates:3 * num_gates] * len(obj_wires)).astype(int)
    cnot_wires = rng.choice(list(np.arange(num_qubits)), size=(num_gates, 2), replace=True)

    gate_choices = [pauli_gates[i] for i in gate_indices]
    wire_choices = [obj_wires[i] for i in wire_indices]

    circuit_structure = []

    for i in range(num_gates):
        if random_choices[i] < ratio_imprim:
            circuit_structure.append({"gate": "CNOT", "wires": list(cnot_wires[i])})
        else:
            circuit_structure.append({"gate": gate_choices[i], "wires": [wire_choices[i]]})
    
    return circuit_structure

def save_circuit_structure_to_json(circuit_structure, filename):
    with open(filename, 'w') as f:
        json.dump(circuit_structure, f)

def load_circuit_structure_from_json(filename):
    with open(filename, 'r') as f:
        circuit_structure = json.load(f)
    return circuit_structure


device = qml.device("default.qubit", wires=args.num_wires, shots=args.num_shots)
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
neural_fn = neural_function(args.dimension, args.hidden_layer)
param_init = np.random.random(qml.RandomLayers.shape(n_layers=args.num_layers, n_rotations=3))
cost_func_store = []
if args.load_circuit_structure_path:
    circuit_structure = load_circuit_structure_from_json(args.load_circuit_structure_path)
else:
    circuit_structure = generate_random_circuit_structure(args.qubits, seed=args.seed)

if args.save_circuit_structure:
    save_circuit_structure_to_json(circuit_structure, f"circuit_structure_{time.time()}.json")

# start the training
for epoch in range(1, args.num_of_epochs):
    
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
            result = measure_rho(shifted, circuit_structure, args.qubits)
            for sample in range(args.num_of_samples):
                sample_result_array = np.array([result[q][sample] for q in range(args.dimension)])
                nn_result = neural_fn(torch.from_numpy(sample_result_array))
                forward_sum += nn_result[0].detach().numpy()

      # normalize the forward sum
            forward_sum = forward_sum/args.num_of_samples

      # left shift the parameters
            shifted[i, j] -= np.pi

      # backward evaluation
            backward_sum = 0
            result = measure_rho(shifted, circuit_structure, args.qubits)
            for sample in range(args.num_of_samples):
                sample_result_array = np.array([result[q][sample] for q in range(args.dimension)])
                nn_result = neural_fn(torch.from_numpy(sample_result_array))
                backward_sum += nn_result[0].detach().numpy()

      # normalize the backward sum
            backward_sum = backward_sum/args.num_of_samples
      #print(backward_sum)

      # parameter-shift rule
            gradients[i, j] = - 0.5 * (forward_sum - backward_sum)
    np.save(f"gradients_epoch{epoch}_{time.time()}.npy", gradients)

  # first copy the quantum circuit parameters before updating it
    prev_param_init = param_init.copy()

  # update the quantum circuit parameters
    param_init -= args.learning_rate*gradients

  # evaluate the gradient with respect to the NN parameters
    optimizer = optim.SGD(neural_fn.parameters(), lr=args.learning_rate)

  # evaluate the first term
    loss = 0
    result = measure_rho(prev_param_init, circuit_structure, args.qubits)
    for sample in range(args.num_of_samples):
        # optimizer.zero_grad()
        sample_result_array = np.array([result[q][sample] for q in range(args.dimension)])
        random_result_array = np.random.choice([-1, 1], size=args.dimension)
        sample_nn_result = neural_fn(torch.from_numpy(sample_result_array))
        random_nn_result = neural_fn(torch.from_numpy(random_result_array))
        loss_term = (torch.exp(random_nn_result[0]) - sample_nn_result[0]).to("cpu")
        loss += loss_term / args.num_of_samples
    loss.backward()
    torch.nn.utils.clip_grad_norm_(neural_fn.parameters(), max_norm=1.0)
    optimizer.step()

  # evaluate the cost function at these parameters
    first_term = 0
    result = measure_rho(param_init, circuit_structure, args.qubits)
    for sample in range(args.num_of_samples):
        sample_result_array = np.array([result[q][sample] for q in range(args.dimension)])
        nn_result = neural_fn(torch.from_numpy(sample_result_array))
        first_term += nn_result[0].detach().numpy()

  # normalize the cost sum
    first_term = first_term/args.num_of_samples

  # # Second term evaluation
    second_term = 0
    for sample in range(args.num_of_samples):
        result = np.random.choice([-1, 1], size=args.dimension)
        nn_result = neural_fn(torch.from_numpy(result.flatten()))
        second_term += np.exp(nn_result[0].detach().numpy())

  # normalize the second term sum
    second_term = second_term/args.num_of_samples

    # add the cost function to the store
    cost_func_store.append(np.log(args.N) - first_term + second_term - 1)

  # print the cost
    print(f"Epoch {epoch}: Loss: {loss.item()} Cost: {np.log(args.N) - first_term + second_term - 1}")