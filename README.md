# Pantheon v0.6 — Associative Memory Test Rig

This release preserves v0.5's live observatory and adds `brain/associative_memory.py`, a separate, *small* functional-memory experiment. We deliberately do not claim the live 256-neuron brain now remembers words or patterns.

## Install

Unzip, copy the **contents** of `pantheon_v06` into your existing `~/pantheon` folder, replacing matching files. Keep your `.venv/` and saved `memory/` data. The added experiment only needs NumPy, which is already installed.

## Run the live observatory

```bash
cd ~/pantheon
source .venv/bin/activate
python main.py
```

## Run Experiment 003 — associative memory

```bash
cd ~/pantheon
source .venv/bin/activate
python -m brain.associative_memory
```

This uses two cue channels and two leaky integrate-and-fire output neurons. Initially a train of A input spikes cannot make the A output neuron spike. During training, A cue activity is followed by an externally imposed A teaching spike, 2 ms later. A causal spike-timing rule strengthens the relevant excitatory weight. At test, teaching and plasticity are disabled: only the cue can activate an output. Ten jittered trials compare baseline, trained, and identical no-learning control. Pattern B is an untrained comparison.

**Important limits:** This is a *supervised, highly simplified* learning circuit **independent of the running 256-neuron Brain**. It demonstrates an elementary synaptic association and neuron-mediated output under controlled conditions. It does **not** show the existing Pantheon brain has functional memory, or show persistent long-term retention, language, or consciousness. The next stage is to incorporate the learned cue-response pathway inside Pantheon's live network and validate it across delays and random seeds. The numeric output here is reproducible by design; it is not an independent biological discovery.

## Git

```bash
git add .
git commit -m "Add associative memory microcircuit experiment"
git push
```
