"""Controlled Pantheon memory assay, run in a separate Python process.

Usage: python -m brain.experiment
Prints JSON on the final stdout line for the GUI to display.
"""
import json
import tempfile
from pathlib import Path
import numpy as np
from brian2 import ms, mV
from .simulation import Brain, N, SEED


def snapshot_test(brain, checkpoint, pattern):
    """Probe from an identical saved state with learning disabled."""
    brain.load(checkpoint)
    brain.set_learning(False)
    start = np.asarray(brain.spikes.count[:], dtype=int)
    brain.stimulate(pattern)
    brain.step(100)
    response = np.asarray(brain.spikes.count[:], dtype=int) - start
    driven = np.arange(12, 28) if pattern == 'A' else np.arange(84, 100)
    response[driven] = 0  # do not mistake directly stimulated cells for recall
    return response


def train_condition(brain, base, destination, plasticity):
    brain.load(base)
    brain.set_learning(plasticity)
    before = brain.weights.copy()
    for _ in range(20):
        brain.stimulate('A')
        brain.step(40)
    brain.step(300)  # no input, common washout duration
    after = brain.weights.copy()
    brain.save(destination)
    return {
        'mean_weight_before_mV': float(np.mean(before)),
        'mean_weight_after_mV': float(np.mean(after)),
        'mean_abs_weight_change_mV': float(np.mean(np.abs(after-before))),
        'max_abs_weight_change_mV': float(np.max(np.abs(after-before))),
    }


def summarize_response(resp):
    return {'spikes_undriven': int(resp.sum()), 'active_undriven_neurons': int(np.count_nonzero(resp))}


def run_experiment():
    brain = Brain()
    with tempfile.TemporaryDirectory(prefix='pantheon_test_') as temp:
        root = Path(temp)
        base, trained, control = (str(root / name) for name in ('base.pkl', 'trained.pkl', 'control.pkl'))
        brain.step(1000)  # identical baseline after settling
        brain.save(base)
        base_a = snapshot_test(brain, base, 'A')
        base_b = snapshot_test(brain, base, 'B')
        learning_stats = train_condition(brain, base, trained, True)
        control_stats = train_condition(brain, base, control, False)
        learned_a = snapshot_test(brain, trained, 'A')
        learned_b = snapshot_test(brain, trained, 'B')
        control_a = snapshot_test(brain, control, 'A')
        control_b = snapshot_test(brain, control, 'B')

    # Basic assay: changing response is not necessarily functional memory.
    result = {
        'seed': SEED,
        'training': '20 pulses of A at 40 ms spacing; 300 ms post-training without input',
        'probe': '100 ms after one pulse; driven neurons excluded; plasticity OFF for all probes',
        'baseline': {'A': summarize_response(base_a), 'B': summarize_response(base_b)},
        'plasticity_on': {'weights': learning_stats, 'A': summarize_response(learned_a),
                          'B': summarize_response(learned_b)},
        'plasticity_off_control': {'weights': control_stats, 'A': summarize_response(control_a),
                                   'B': summarize_response(control_b)},
        'response_change': {
            'trained_A_minus_control_A_spikes': int(learned_a.sum() - control_a.sum()),
            'trained_B_minus_control_B_spikes': int(learned_b.sum() - control_b.sum()),
            'trained_A_minus_control_A_L1': int(np.abs(learned_a-control_a).sum()),
            'trained_B_minus_control_B_L1': int(np.abs(learned_b-control_b).sum()),
        },
        'interpretation': 'One deterministic trial only. Weight change confirms plasticity; response differences are not proof of memory. Repeat across seeds and test long retention for stronger evidence.'
    }
    return result


if __name__ == '__main__':
    print('Running isolated memory experiment...', flush=True)
    print(json.dumps(run_experiment(), indent=2))
