"""Pantheon Experiment 003: a minimal spike-timing associative memory.

This independent microcircuit is a deliberately small *test rig*, not the
256-neuron live Brain. It uses leaky integrate-and-fire output neurons and a
causal pre/post spike-timing learning rule. No response is hard-coded.
Run: python -m brain.associative_memory
"""
from dataclasses import dataclass
import json
import numpy as np

DT_MS = 1.0
TAU_MEM_MS = 20.0
TAU_TRACE_MS = 20.0
THRESHOLD = 1.0
RESET = 0.0
INITIAL_WEIGHT = 0.12
MAX_WEIGHT = 0.9
LEARNING_RATE = 0.035
N_TRAIN = 40


@dataclass
class Circuit:
    """Two independent cue inputs A/B, two spiking output cells A/B."""
    weights: np.ndarray

    @classmethod
    def new(cls):
        return cls(np.diag([INITIAL_WEIGHT, INITIAL_WEIGHT]).astype(float))

    def train_a(self, repetitions=N_TRAIN, plasticity=True):
        """Present A cue 2ms before an externally imposed A teacher spike.

        A presynaptic eligibility trace precedes a postsynaptic spike.
        The update is applied ONLY when the learning flag is enabled.
        """
        if not plasticity:
            return
        trace_at_teacher = np.exp(-2.0 / TAU_TRACE_MS)
        for _ in range(repetitions):
            self.weights[0, 0] = np.clip(
                self.weights[0, 0] + LEARNING_RATE * trace_at_teacher,
                0.0, MAX_WEIGHT,
            )

    def probe(self, cue, jitter_seed=0):
        """Run a discrete-time LIF model with cue spikes; no teacher or learning.

        Return spikes from both output neurons, excluding any input spikes.
        The randomized cue timing makes this a repeatable test of robustness.
        """
        rng = np.random.default_rng(jitter_seed)
        cue_index = {'A': 0, 'B': 1}[cue]
        # Ten presynaptic spikes across a 100ms trial, +/-1ms jitter.
        spike_times = np.arange(10, 90, 8) + rng.integers(-1, 2, size=10)
        voltage = np.zeros(2, dtype=float)
        output_counts = np.zeros(2, dtype=int)
        voltage_decay = np.exp(-DT_MS / TAU_MEM_MS)
        for t in range(100):
            voltage *= voltage_decay
            if np.any(spike_times == t):
                voltage += self.weights[cue_index, :]
            fired = voltage >= THRESHOLD
            output_counts += fired.astype(int)
            voltage[fired] = RESET
        return output_counts


def experiment():
    baseline = Circuit.new()
    trained = Circuit.new()
    control = Circuit.new()
    trained.train_a(plasticity=True)
    control.train_a(plasticity=False)
    conditions = {'baseline': baseline, 'trained': trained, 'control': control}
    results = {}
    for name, circuit in conditions.items():
        results[name] = {
            'weights': np.round(circuit.weights, 6).tolist(),
            'A': [circuit.probe('A', seed).tolist() for seed in range(10)],
            'B': [circuit.probe('B', seed).tolist() for seed in range(10)],
        }
    a_trained = np.array(results['trained']['A'])[:, 0]
    a_control = np.array(results['control']['A'])[:, 0]
    b_trained = np.array(results['trained']['B'])[:, 1]
    b_control = np.array(results['control']['B'])[:, 1]
    results['analysis'] = {
        'A_target_mean_spikes_trained': float(a_trained.mean()),
        'A_target_mean_spikes_control': float(a_control.mean()),
        'B_target_mean_spikes_trained': float(b_trained.mean()),
        'B_target_mean_spikes_control': float(b_control.mean()),
        'A_gain_over_control': float((a_trained - a_control).mean()),
        'B_gain_over_control': float((b_trained - b_control).mean()),
        'trained_A_weight_change': float(trained.weights[0, 0] - baseline.weights[0, 0]),
        'control_A_weight_change': float(control.weights[0, 0] - baseline.weights[0, 0]),
        'passes_basic_association_test': bool(
            (a_trained - a_control).mean() > 0 and
            abs((b_trained - b_control).mean()) < 1e-9
        ),
    }
    return results


def main():
    outcome = experiment()
    print('PANTHEON EXPERIMENT 003 — ASSOCIATIVE MEMORY')
    print('Architecture: 2 cue channels → 2 leaky integrate-and-fire output neurons')
    print('Training: 40 A cue / A teacher pairs, 2ms apart; causal spike-timing plasticity')
    print('Testing: 10 deterministic jittered trials per cue, no teacher, plasticity off')
    for label in ('baseline', 'trained', 'control'):
        x = outcome[label]
        a = np.array(x['A'])[:, 0]
        b = np.array(x['B'])[:, 1]
        print(f'{label:9s} A→A output spikes: {a.mean():.2f}  B→B output spikes: {b.mean():.2f}  weights: {x["weights"]}')
    print('A improvement over no-learning control:', outcome['analysis']['A_gain_over_control'])
    print('B improvement over no-learning control:', outcome['analysis']['B_gain_over_control'])
    print('Basic association test passed:', outcome['analysis']['passes_basic_association_test'])
    print('\nLIMITATION: This is a separate, teacher-guided microcircuit, not memory in the live 256-neuron brain.')
    print('It tests cue-response association, not autonomous cognition or semantic memory.')
    print('\nJSON RESULTS:')
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    main()
