# Pantheon v0.7 — Integrated associative memory

The existing 256-neuron Brian2 network now contains two *real*, plastic associative pathways linking cue neurons to output neurons. The organic observatory, zoom, speed control, save/load, and click-to-inspect remain.

This is a teacher-guided associative-memory experiment, **not** language, awareness, or autonomous cognition. It tests whether plastic synapses inside the recurrent network influence later responses. A result is not guaranteed.

## Install

Copy all files in this folder into `~/pantheon`, replacing existing files. Keep `~/pantheon/.venv` and any personal `memory/` checkpoints. v0.7 changes network structure: **older v0.4–v0.6 checkpoints may not restore in v0.7**. Keep backups of them.

```bash
cd ~/pantheon
source .venv/bin/activate
python main.py
```

## Experiment 004

In the observatory press **Experiment 004: Retention**. It runs a *quick* isolated experiment in another process. Or use the terminal:

```bash
python -m brain.retention
```

The quick test checks memory after 1, 3, and 10 simulated seconds. A fuller, potentially much slower test uses 10, 60, and 300 seconds:

```bash
python -m brain.retention --full
```

The experiment builds a fresh 256-cell network with two 16-cell cue populations (A: neurons 12–27; B: 84–99) and two 16-cell response populations (A: 40–55; B: 112–127). It adds 512 real, learnable synapses between existing cells. Each branch starts from the same neural/random state. The trained branch receives 40 cue A / teacher A pairs with plasticity enabled. The matched control receives the same pulses with plasticity disabled. Both continue evolving during the retention delay, and each cue is probed from matched within-branch snapshots with learning turned off during the probe.

Crucial limitation: this is a *designed* associative pathway, with a teaching signal; the network did not autonomously discover which neurons should represent A or B. The experiment does not run multiple random seeds, and exact recall performance remains to be tested on your Mac.

## Practical notes

- Each simulation-second is numerical simulation time, not necessarily one wall-clock second. The 300-second run can be slow.
- Checkpoints include Brian2 simulation objects and should only be loaded from trusted sources. Existing checkpoints are not guaranteed compatible after adding new synapses.
- The visual synapse lines are real connections but their spatial arrangement is illustrative.
- The display reads neural state; it does not influence the dynamics.
