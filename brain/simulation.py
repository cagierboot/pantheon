"""Pantheon's stateful Brian2 spiking neural network."""
from pathlib import Path
import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network, start_scope, seed, prefs,
    ms, mV, second,
)

N = 256
N_EXC = 205
SEED = 41


class Brain:
    def __init__(self):
        start_scope()
        seed(SEED)  # Reproduce wiring on restart so saved states can be restored.
        prefs.codegen.target = 'numpy'
        eqs = '''
        dv/dt = (v_rest - v)/tau + sigma*xi/sqrt(tau) : volt
        v_rest : volt (constant)
        tau : second (constant)
        sigma : volt (constant)
        '''
        self.neurons = NeuronGroup(
            N, eqs, threshold='v > -50*mV', reset='v = -70*mV',
            refractory=3*ms, method='euler', name='neurons',
        )
        self.neurons.v_rest = -65*mV
        self.neurons.tau = 20*ms
        self.neurons.sigma = 8*mV
        self.neurons.v = '-65*mV + randn()*5*mV'

        # Pair-based STDP; bounded excitatory weights, no gradients or language model.
        self.exc = Synapses(
            self.neurons[:N_EXC], self.neurons,
            model='''
            w : volt
            learning_enabled : 1 (shared)
            dapre/dt = -apre/(20*ms) : 1 (event-driven)
            dapost/dt = -apost/(20*ms) : 1 (event-driven)
            ''',
            on_pre='''
            v_post += w
            apre += 0.01
            w = clip(w + learning_enabled*apost*0.1*mV, 0.05*mV, 1.2*mV)
            ''',
            on_post='''
            apost -= 0.0105
            w = clip(w + learning_enabled*apre*0.1*mV, 0.05*mV, 1.2*mV)
            ''',
            name='excitatory_synapses',
        )
        self.exc.connect(p=0.06)
        self.exc.w = '0.3*mV + rand()*0.4*mV'
        self.exc.learning_enabled = 1
        self.inh = Synapses(
            self.neurons[N_EXC:], self.neurons,
            on_pre='v_post -= 1.5*mV', name='inhibitory_synapses',
        )
        self.inh.connect(p=0.06)
        self.spikes = SpikeMonitor(self.neurons, name='spikes')
        self.net = Network(self.neurons, self.exc, self.inh, self.spikes)
        self.read_index = 0

        self.exc_sources = np.asarray(self.exc.i[:], dtype=int)
        self.exc_targets = np.asarray(self.exc.j[:], dtype=int)
        self.inh_sources = np.asarray(self.inh.i[:], dtype=int) + N_EXC
        self.inh_targets = np.asarray(self.inh.j[:], dtype=int)
        self.edges = ([(int(i), int(j), True) for i, j in zip(self.exc_sources, self.exc_targets)]
                      + [(int(i), int(j), False) for i, j in zip(self.inh_sources, self.inh_targets)])

    def step(self, duration_ms=20):
        self.net.run(float(duration_ms) * ms)
        end = self.spikes.num_spikes
        ids = np.asarray(self.spikes.i[self.read_index:end], dtype=int).copy()
        self.read_index = end
        return ids

    def stimulate(self, pattern='A'):
        # Controlled artificial stimulation, not a word or semantic representation.
        ids = np.arange(12, 28) if pattern == 'A' else np.arange(84, 100)
        voltages = np.asarray(self.neurons.v[:] / mV).copy()
        voltages[ids] += 18.0
        self.neurons.v = voltages * mV

    def set_learning(self, enabled):
        self.exc.learning_enabled = int(enabled)

    @property
    def learning(self):
        return bool(int(np.asarray(self.exc.learning_enabled[:]).item()))

    @property
    def time(self):
        return float(self.net.t / second)

    @property
    def weights(self):
        return np.asarray(self.exc.w[:] / mV, dtype=float)

    def inspect(self, neuron_id):
        i = int(neuron_id)
        incoming = int(np.count_nonzero(self.exc_targets == i) + np.count_nonzero(self.inh_targets == i))
        outgoing = int(np.count_nonzero(self.exc_sources == i) + np.count_nonzero(self.inh_sources == i))
        return {'voltage': float(self.neurons.v[i] / mV), 'incoming': incoming,
                'outgoing': outgoing, 'spikes': int(self.spikes.count[i]),
                'kind': 'Excitatory' if i < N_EXC else 'Inhibitory'}

    def save(self, filename):
        # Brian2 saves membrane voltages, synaptic state, event traces, monitors,
        # clock state and (for NumPy codegen) random-generator state.
        self.net.store('checkpoint', filename=str(Path(filename)))

    def load(self, filename):
        self.net.restore('checkpoint', filename=str(Path(filename)), restore_random_state=True)
        self.read_index = self.spikes.num_spikes
