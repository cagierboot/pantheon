"""Experiment 004: associative recall inside Pantheon's 256-neuron Brian2 network.

Quick: python -m brain.retention
Full:  python -m brain.retention --full

Training pairs cue A followed by a teacher A signal. Tests cue-only recall
at multiple delays versus an identical learning-disabled branch. Does not
modify the live observer's Brain instance.
"""
import argparse
import numpy as np
from brian2 import ms, mV
from .simulation import Brain


def response(brain, cue):
    """Counts target spikes after a single cue, excluding directly driven cells."""
    # Both branches and both cues start from the same stored state.
    brain.set_learning(False)
    initial = np.asarray(brain.spikes.count[:], dtype=int)
    brain.cue(cue)
    brain.step(1)
    brain.step(39)
    delta = np.asarray(brain.spikes.count[:], dtype=int) - initial
    target = slice(40, 56) if cue == 'A' else slice(112, 128)
    other = slice(112, 128) if cue == 'A' else slice(40, 56)
    return int(delta[target].sum()), int(delta[other].sum())


def train(brain, plasticity, n=40):
    brain.set_learning(plasticity)
    for _ in range(n):
        brain.cue('A')
        brain.step(2)
        brain.teacher('A')
        brain.step(38)


def one_branch(brain, plasticity, delays):
    brain.net.restore('initial', restore_random_state=True)
    brain.read_index = brain.spikes.num_spikes
    brain.set_learning(plasticity)
    before = np.asarray(brain.assoc.w[:]/mV).copy()
    train(brain, plasticity)
    after = np.asarray(brain.assoc.w[:]/mV).copy()
    brain.net.store('post_training')
    results = {}
    elapsed = 0.0
    for delay in delays:
        # Advance freely with spontaneous activity; keep the same learning
        # setting as the training branch. Both arms receive equal idle time.
        brain.net.restore('post_training', restore_random_state=True)
        brain.set_learning(plasticity)
        if delay:
            brain.step(delay * 1000)
        brain.net.store('before_probe')
        results[delay] = {}
        for cue in ('A', 'B'):
            brain.net.restore('before_probe', restore_random_state=True)
            results[delay][cue] = response(brain, cue)
    return before, after, results


def run(full=False):
    delays = [10, 60, 300] if full else [1, 3, 10]
    b = Brain()
    # Quiet memory readout cells retain their integration behavior but are not
    # directly hit by the independent background noise term.
    # They still receive recurrent synapses from the full brain.
    b.neurons.sigma[40:56] = 0*mV
    b.neurons.sigma[112:128] = 0*mV
    b.net.store('initial')
    trained_before, trained_after, trained = one_branch(b, True, delays)
    control_before, control_after, control = one_branch(b, False, delays)
    assert np.allclose(trained_before, control_before)
    a_idx = np.arange(len(trained_before)) < 256
    print('PANTHEON EXPERIMENT 004 — INTEGRATED ASSOCIATIVE MEMORY')
    print('Circuit: two 16-neuron cue groups and two 16-neuron response groups within a 256-neuron recurrent network')
    print('Training: 40 paired A cue / A teacher trials; 2 ms cue-to-teacher gap')
    print('Probes: cue only; plasticity disabled during probes; matched random states per branch')
    print('Note: response populations have no independent background noise but receive recurrent input.')
    print(f'Mode: {"full" if full else "quick"} | idle delays: {delays} simulated seconds')
    print(f'A associative weights (mean): initial {trained_before[:256].mean():.4f} mV, '
          f'trained {trained_after[:256].mean():.4f} mV, control {control_after[:256].mean():.4f} mV')
    print('\n        TRAINED (target/off-target)           CONTROL (target/off-target)')
    for d in delays:
        for cue in ('A', 'B'):
            x = trained[d][cue]; y = control[d][cue]
            print(f'{d:>3}s {cue}     {x[0]:>4}/{x[1]:<4}                           {y[0]:>4}/{y[1]:<4}')
    print('\nInterpretation: Compare trained A against control A AND trained B.')
    print('A positive trained A-control A difference alone is not conclusive; repeat across seeds and longer delays.')
    print('This uses teacher-guided training and predefined populations. It is not semantic memory or consciousness.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Test retention at 10, 60 and 300 simulated seconds (slow).')
    args = parser.parse_args()
    run(args.full)
