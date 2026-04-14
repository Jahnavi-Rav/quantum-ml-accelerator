import numpy as np
import scipy.optimize as opt
from typing import List, Tuple, Callable
import networkx as nx
from dataclasses import dataclass

@dataclass
class QAOAParams:
    """QAOA circuit parameters"""
    betas: np.ndarray   # Mixer angles
    gammas: np.ndarray  # Cost angles
    p: int  # Number of QAOA layers

class QuantumInspiredOptimizer:
    """Quantum-Approximate Optimization Algorithm (QAOA) for combinatorial problems"""
    
    def __init__(self, num_qubits: int, p: int = 3):
        self.num_qubits = num_qubits
        self.p = p
        self.best_params = None
        self.best_energy = float('inf')
        
    def initialize_state(self) -> np.ndarray:
        """Initialize quantum state in equal superposition"""
        dim = 2 ** self.num_qubits
        return np.ones(dim) / np.sqrt(dim)
    
    def cost_hamiltonian(self, state: np.ndarray, problem_graph: nx.Graph) -> float:
        """Compute expectation value of cost Hamiltonian (MaxCut problem)"""
        energy = 0.0
        dim = len(state)
        
        for i, j in problem_graph.edges():
            # For each edge, add contribution if nodes have different spin
            for bitstring in range(dim):
                bit_i = (bitstring >> i) & 1
                bit_j = (bitstring >> j) & 1
                
                if bit_i != bit_j:
                    energy += np.abs(state[bitstring]) ** 2
                    
        return energy
    
    def mixer_hamiltonian(self, state: np.ndarray, beta: float) -> np.ndarray:
        """Apply mixer Hamiltonian (X-rotations on all qubits)"""
        dim = len(state)
        new_state = np.zeros(dim, dtype=complex)
        
        for bitstring in range(dim):
            for qubit in range(self.num_qubits):
                # Flip qubit
                flipped = bitstring ^ (1 << qubit)
                new_state[flipped] += -1j * np.sin(beta) * state[bitstring]
                new_state[bitstring] += np.cos(beta) * state[bitstring]
                
        return new_state / np.sqrt(self.num_qubits)
    
    def cost_layer(self, state: np.ndarray, gamma: float, 
                   problem_graph: nx.Graph) -> np.ndarray:
        """Apply cost Hamiltonian layer (ZZ interactions)"""
        dim = len(state)
        new_state = state.copy()
        
        for i, j in problem_graph.edges():
            for bitstring in range(dim):
                bit_i = (bitstring >> i) & 1
                bit_j = (bitstring >> j) & 1
                
                # Apply phase based on edge
                phase = np.exp(-1j * gamma) if bit_i == bit_j else np.exp(1j * gamma)
                new_state[bitstring] *= phase
                
        return new_state
    
    def qaoa_circuit(self, params: QAOAParams, 
                     problem_graph: nx.Graph) -> np.ndarray:
        """Execute full QAOA circuit"""
        state = self.initialize_state()
        
        for layer in range(self.p):
            # Apply cost layer
            state = self.cost_layer(state, params.gammas[layer], problem_graph)
            # Apply mixer layer
            state = self.mixer_hamiltonian(state, params.betas[layer])
            
        return state
    
    def objective_function(self, flat_params: np.ndarray, 
                          problem_graph: nx.Graph) -> float:
        """Objective function for classical optimization"""
        betas = flat_params[:self.p]
        gammas = flat_params[self.p:]
        
        params = QAOAParams(betas=betas, gammas=gammas, p=self.p)
        final_state = self.qaoa_circuit(params, problem_graph)
        
        return -self.cost_hamiltonian(final_state, problem_graph)
    
    def optimize(self, problem_graph: nx.Graph, 
                max_iter: int = 100) -> Tuple[QAOAParams, float]:
        """Optimize QAOA parameters using classical optimizer"""
        # Initialize parameters randomly
        initial_params = np.random.uniform(0, 2*np.pi, 2*self.p)
        
        # Run optimization
        result = opt.minimize(
            lambda p: self.objective_function(p, problem_graph),
            initial_params,
            method='COBYLA',
            options={'maxiter': max_iter}
        )
        
        # Extract optimal parameters
        opt_betas = result.x[:self.p]
        opt_gammas = result.x[self.p:]
        
        self.best_params = QAOAParams(betas=opt_betas, gammas=opt_gammas, p=self.p)
        self.best_energy = -result.fun
        
        return self.best_params, self.best_energy
    
    def sample_solution(self, params: QAOAParams, 
                       problem_graph: nx.Graph, 
                       num_shots: int = 1000) -> dict:
        """Sample bitstrings from final state"""
        final_state = self.qaoa_circuit(params, problem_graph)
        probabilities = np.abs(final_state) ** 2
        
        # Sample bitstrings
        bitstrings = np.random.choice(
            len(final_state), 
            size=num_shots, 
            p=probabilities
        )
        
        # Count occurrences
        counts = {}
        for bitstring in bitstrings:
            bit_str = format(bitstring, f'0{self.num_qubits}b')
            counts[bit_str] = counts.get(bit_str, 0) + 1
            
        return counts

class VQEOptimizer:
    """Variational Quantum Eigensolver for finding ground states"""
    
    def __init__(self, num_qubits: int, num_layers: int = 3):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_params = num_qubits * num_layers * 3  # 3 rotation angles per qubit per layer
        
    def ansatz(self, params: np.ndarray) -> np.ndarray:
        """Hardware-efficient ansatz with parameterized rotations"""
        dim = 2 ** self.num_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0  # Start in |0...0>
        
        param_idx = 0
        for layer in range(self.num_layers):
            # Single-qubit rotations
            for qubit in range(self.num_qubits):
                theta = params[param_idx]
                phi = params[param_idx + 1]
                lamb = params[param_idx + 2]
                state = self._apply_u3(state, qubit, theta, phi, lamb)
                param_idx += 3
            
            # Entangling layer (CNOT ladder)
            for qubit in range(self.num_qubits - 1):
                state = self._apply_cnot(state, qubit, qubit + 1)
                
        return state
    
    def _apply_u3(self, state: np.ndarray, qubit: int, 
                  theta: float, phi: float, lamb: float) -> np.ndarray:
        """Apply U3 gate (general single-qubit rotation)"""
        # Simplified implementation
        return state  # Placeholder
    
    def _apply_cnot(self, state: np.ndarray, 
                    control: int, target: int) -> np.ndarray:
        """Apply CNOT gate"""
        # Simplified implementation
        return state  # Placeholder
    
    def compute_energy(self, params: np.ndarray, 
                      hamiltonian: callable) -> float:
        """Compute energy expectation value"""
        state = self.ansatz(params)
        return hamiltonian(state)
    
    def optimize(self, hamiltonian: callable, 
                max_iter: int = 200) -> Tuple[np.ndarray, float]:
        """Find ground state energy"""
        initial_params = np.random.uniform(0, 2*np.pi, self.num_params)
        
        result = opt.minimize(
            lambda p: self.compute_energy(p, hamiltonian),
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter}
        )
        
        return result.x, result.fun

# Example usage
if __name__ == "__main__":
    # Create MaxCut problem graph
    G = nx.erdos_renyi_graph(n=6, p=0.5, seed=42)
    
    # Initialize QAOA optimizer
    qaoa = QuantumInspiredOptimizer(num_qubits=6, p=3)
    
    # Optimize
    params, energy = qaoa.optimize(G, max_iter=100)
    print(f"Optimal energy: {energy}")
    
    # Sample solutions
    counts = qaoa.sample_solution(params, G, num_shots=1000)
    print(f"Top solutions: {sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
