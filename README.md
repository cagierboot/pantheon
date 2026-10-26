# Pantheon v0.4 — Neural Development

A local, experimental 256-neuron spiking neural network with Brian2 and a Qt observatory.

## Run on the existing Mac environment

```bash
cd ~/pantheon
source .venv/bin/activate
python main.py
```

If you need to install the pinned dependencies: `python -m pip install -r requirements.txt`.
These versions target your working Python 3.11 / macOS 14.3 installation.

## Controls

- Mouse wheel: zoom; drag: pan; double-click: reset; click a neuron: inspect.
- Speed dropdown: simulated milliseconds per UI update (not guaranteed real-time speed).
- Learning enabled: toggle excitatory STDP.
- Pulse pattern A/B: directly stimulate different groups of neurons.
- Train A ×20: repeated A stimulation, while simulation progresses.
- Save brain / Load brain: Brian2 snapshot file. Treat `.pkl` snapshots as **trusted local files only**; never load unknown snapshots.

## Scientific limitations

- Node placement is an illustrative projection. Synaptic lines show actual directed connections, but line glows indicate recent presynaptic spikes, not measured pulse transit.
- Spontaneous firing partly results from artificial stochastic drive.
- STDP modifies excitatory synapses; it is not evidence of concepts, consciousness, or useful learning.
- To establish learning, make reproducible before/after response measurements and a plasticity-disabled control experiment. This UI is not yet a formal learning benchmark.
- The SpikeMonitor keeps spikes for the duration of a session, so long runs increase memory use.
- Saving persists simulation state, not wall-clock activity while the app is shut down.
# pantheon
