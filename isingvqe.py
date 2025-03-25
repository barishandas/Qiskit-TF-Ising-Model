import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from scipy.optimize import minimize
from functools import reduce
from itertools import combinations

class LargeIsingVQE:
    """
    A Variational Quantum Eigensolver (VQE) for larger Ising Model systems
    with arbitrary number of qubits and configurable connectivity.
    """
    
    def __init__(self, num_qubits, J=1.0, h=1.0, connectivity='linear', shots=1024):
        """
        Initialize the simulation
        
        Args:
            num_qubits (int): Number of qubits in the system
            J (float): Interaction strength between qubits (Z-Z coupling)
            h (float): Transverse field strength (X component)
            connectivity (str): Type of connectivity ('linear', 'full', or 'custom')
            shots (int): Number of measurement shots
        """
        self.num_qubits = num_qubits
        self.J = J
        self.h = h
        self.shots = shots
        self.connectivity = connectivity
        self.backend = Aer.get_backend('statevector_simulator')
        self.energy_history = []
        
        # Define connectivity graph based on type
        self.connections = self._create_connectivity()
        
        # Calculate the Hamiltonian matrix
        self.hamiltonian = self._create_hamiltonian()
        
        # Calculate exact eigenvalues (for small enough systems)
        if num_qubits <= 10:  # Limit for exact diagonalization
            try:
                self.exact_eigenvalues = np.linalg.eigvalsh(self.hamiltonian)
                self.exact_ground_energy = min(self.exact_eigenvalues)
                print(f"Exact ground state energy: {self.exact_ground_energy}")
            except:
                print("System too large for exact diagonalization")
                self.exact_ground_energy = None
        else:
            print("System too large for exact diagonalization")
            self.exact_ground_energy = None

    def _create_connectivity(self):
        """Create the connectivity graph for the Ising model"""
        connections = []
        if self.connectivity == 'linear':
            # 1D chain with nearest-neighbor interactions
            connections = [(i, i+1) for i in range(self.num_qubits-1)]
        elif self.connectivity == 'full':
            # All-to-all connectivity
            connections = list(combinations(range(self.num_qubits), 2))
        # Add custom connectivity options here if needed
        return connections

    def _create_hamiltonian(self):
        """
        Create the Hamiltonian matrix for the larger Ising Model
        H = -J * Σ Z_i Z_j - h * Σ X_i
        """
        dim = 2**self.num_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        # Helper function for tensor product of Pauli matrices
        def create_pauli_string(op_positions, op_type):
            ops = []
            for i in range(self.num_qubits):
                if i in op_positions:
                    ops.append(op_type)
                else:
                    ops.append(np.eye(2))
            return reduce(np.kron, ops)
        
        # Add ZZ interactions based on connectivity
        for i, j in self.connections:
            zz_term = create_pauli_string([i, j], np.array([[1, 0], [0, -1]]))
            H -= self.J * zz_term
        
        # Add transverse field terms (X)
        for i in range(self.num_qubits):
            x_term = create_pauli_string([i], np.array([[0, 1], [1, 0]]))
            H -= self.h * x_term
        
        return H

    def create_ansatz(self, params):
        """
        Create a hardware-efficient ansatz for the larger system
        
        Args:
            params: List of parameters for the circuit
            
        Returns:
            QuantumCircuit: Parameterized quantum circuit
        """
        circuit = QuantumCircuit(self.num_qubits)
        
        # Calculate number of layers from parameters
        num_layers = len(params) // (2 * self.num_qubits)
        param_idx = 0
        
        for layer in range(num_layers):
            # Rotation layer
            for qubit in range(self.num_qubits):
                circuit.ry(params[param_idx], qubit)
                param_idx += 1
                circuit.rz(params[param_idx], qubit)
                param_idx += 1
            
            # Entangling layer based on connectivity
            for i, j in self.connections:
                circuit.cx(i, j)
        
        return circuit

    def visualize_ansatz(self, params=None):
        """
        Visualize the quantum circuit ansatz
        
        Args:
            params: Optional parameters to display in the circuit
        """
        if params is None:
            # Use random parameters for visualization
            params = np.random.random(6) * 2 * np.pi
            
        circuit = self.create_ansatz(params)
        print("Quantum Circuit Ansatz:")
        print(circuit.draw())
        
        return circuit
    
    def get_expectation(self, params):
        """Calculate the energy expectation value"""
        circuit = self.create_ansatz(params)
        transpiled_circuit = transpile(circuit, self.backend)
        state = Statevector.from_instruction(transpiled_circuit)
        statevector = state.data
        energy = np.real(np.vdot(statevector, np.dot(self.hamiltonian, statevector)))
        self.energy_history.append(energy)
        return energy

    def optimize(self, initial_params=None, method='COBYLA', max_iter=200):
        """Optimize the circuit parameters"""
        # Number of parameters needed for the ansatz
        num_layers = 3  # Can be adjusted
        num_params = num_layers * 2 * self.num_qubits
        
        if initial_params is None:
            initial_params = np.random.random(num_params) * 2 * np.pi
        
        self.energy_history = []
        
        print("Starting VQE optimization...")
        result = minimize(
            self.get_expectation,
            initial_params,
            method=method,
            options={'maxiter': max_iter}
        )
        
        print(f"Optimization completed!")
        print(f"Final energy: {result.fun}")
        if self.exact_ground_energy is not None:
            print(f"Error from exact: {abs(result.fun - self.exact_ground_energy)}")
        
        return {
            'optimal_params': result.x,
            'optimal_energy': result.fun,
            'error': abs(result.fun - self.exact_ground_energy) if self.exact_ground_energy is not None else None,
            'iterations': len(self.energy_history),
            'success': result.success
        }

    def plot_convergence(self):
        """Plot the energy convergence during optimization"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.energy_history, label='VQE Energy')
        if self.exact_ground_energy is not None:
            plt.axhline(y=self.exact_ground_energy, color='r', linestyle='--', 
                       label=f'Exact Ground Energy: {self.exact_ground_energy:.4f}')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title(f'VQE Energy Convergence ({self.num_qubits} qubits)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage
def main_large():
    np.random.seed(42)
    
    # Create a larger Ising model (e.g., 6 qubits)
    num_qubits = 8
    vqe = LargeIsingVQE(
        num_qubits=num_qubits,
        J=6.0,
        h=3.0,
        connectivity='linear',
        shots=1024
    )
    
    vqe.visualize_ansatz()
    
    # Run optimization
    results = vqe.optimize(max_iter=2000)
    
    # Plot convergence
    vqe.plot_convergence()
    
    return vqe, results

if __name__ == "__main__":
    vqe, results = main_large()
