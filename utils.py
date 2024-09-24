import json
import numpy as np
import scipy as sc

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# 補助qubitの変更の加味
def calculate_entropy(rho, N):
    sigma = np.eye(N) / N
    np.seterr(divide='ignore', invalid='ignore')
    H_rho = np.real(np.trace(rho @ (sc.linalg.logm(rho) - sc.linalg.logm(sigma))))
    np.seterr(divide = 'warn', invalid='warn')
    return H_rho

def generate_random_circuit_structure(qubits, num_wires, ratio_imprim=0.7, pauli_gates=['PauliX', 'PauliY', 'PauliZ'], seed=None):
    if seed:
        np.random.seed(seed)

    obj_wires = range(qubits)
    num_qubits = num_wires
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
        # デバッグ用に print で確認する
        print(f"Random choice: {random_choices[i]} (threshold: {ratio_imprim})")
        
        if random_choices[i] < ratio_imprim:
            circuit_structure.append({"gate": "CNOT", "wires": list(cnot_wires[i])})
        else:
            circuit_structure.append({"gate": gate_choices[i], "wires": [wire_choices[i]]})

    return circuit_structure

def save_circuit_structure_to_json(circuit_structure, filename):
    def convert_types(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filename, 'w') as f:
        # defaultパラメータを使用して、numpy型を変換
        json.dump(circuit_structure, f, default=convert_types, indent=4)

def load_circuit_structure_from_json(filename):
    with open(filename, 'r') as f:
        circuit_structure = json.load(f)
    return circuit_structure

def json_to_qml(json_data):
    # ゲートをQML形式に変換するためのリスト
    qml_code = []
    
    gate_map = {
        "CNOT": "qml.CNOT",
        "PauliX": "qml.PauliX",
        "PauliZ": "qml.PauliZ",
        "PauliY": "qml.PauliY"
    }
    
    # 各ゲートデータを処理
    for gate in json_data:
        gate_type = gate.get("gate")
        wires = gate.get("wires")
        
        if gate_type in gate_map:
            # Pennylaneのフォーマットに従って文字列を生成
            qml_code.append(f"{gate_map[gate_type]}(wires={wires})")
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    # 結果をテキスト形式で返す
    return "\n".join(qml_code)