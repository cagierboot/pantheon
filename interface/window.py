"""Pantheon v0.4 observatory, controls, and inspection panel."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QComboBox, QPushButton, QFileDialog,
                               QMessageBox, QCheckBox, QFrame)
from .canvas import NeuralCanvas


class PantheonWindow(QMainWindow):
    def __init__(self, brain):
        super().__init__()
        self.brain = brain
        self.setWindowTitle('PANTHEON v0.4 — Neural Observatory')
        self.resize(1240, 830)
        self.setStyleSheet('''
            QMainWindow, QWidget {background:#080f1f;color:#d7e8f5;}
            QLabel#title {font-size:18px;font-weight:700;}
            QLabel#label {color:#829aaf;font-size:11px;}
            QLabel#value {color:#e5f3ff;font-size:15px;font-weight:600;}
            QPushButton,QComboBox {background:#14243a;border:1px solid #30445d;
                                    border-radius:6px;padding:7px 10px;}
            QPushButton:hover,QComboBox:hover {background:#203853;}
            QFrame#panel {border:1px solid #233852;border-radius:7px;}
        ''')
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18,12,18,12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel('PANTHEON  /  NEURAL OBSERVATORY')
        title.setObjectName('title')
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel('Speed'))
        self.speed = QComboBox()
        for label, value in [('0.25×',.25),('0.5×',.5),('1×',1),('2×',2),('5×',5),('10×',10)]:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(2)
        header.addWidget(self.speed)
        self.pause = QPushButton('Pause')
        self.pause.clicked.connect(self.toggle_pause)
        header.addWidget(self.pause)
        reset = QPushButton('Reset view')
        reset.clicked.connect(lambda: self.canvas.reset_view())
        header.addWidget(reset)
        layout.addLayout(header)

        stats = QHBoxLayout()
        self.stats = {}
        for name, text in [('neurons','NEURONS'),('synapses','SYNAPSES'),
                           ('spikes','SPIKES / STEP'),('time','SIMULATED TIME'),
                           ('weight','MEAN EXC. WEIGHT')]:
            col = QVBoxLayout()
            label = QLabel(text); label.setObjectName('label')
            val = QLabel('—'); val.setObjectName('value')
            col.addWidget(label); col.addWidget(val)
            stats.addLayout(col, 1)
            self.stats[name] = val
        layout.addLayout(stats)

        self.canvas = NeuralCanvas(brain)
        self.canvas.neuron_selected.connect(self.show_neuron)
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        self.learn = QCheckBox('Learning enabled')
        self.learn.setChecked(True)
        self.learn.toggled.connect(self.brain.set_learning)
        controls.addWidget(self.learn)
        for name, pattern in [('Pulse pattern A','A'),('Pulse pattern B','B')]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked=False, p=pattern: self.pulse(p))
            controls.addWidget(btn)
        train = QPushButton('Train A ×20')
        train.clicked.connect(self.train_a)
        controls.addWidget(train)
        save = QPushButton('Save brain')
        save.clicked.connect(self.save_brain)
        controls.addWidget(save)
        load = QPushButton('Load brain')
        load.clicked.connect(self.load_brain)
        controls.addWidget(load)
        layout.addLayout(controls)

        self.detail = QLabel('Click a neuron to inspect it. Pulse A/B stimulate distinct groups; Train A repeats A 20 times.')
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        hint = QLabel('Scroll: zoom  •  Drag: pan  •  Double-click: reset view  •  White flash: real spike')
        hint.setObjectName('label')
        layout.addWidget(hint)
        self.setCentralWidget(root)

        self.stats['neurons'].setText(str(len(brain.neurons)))
        self.stats['synapses'].setText(f'{len(brain.edges):,}')
        self.selected = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(50)
        self.refresh_stats(0)

    def refresh_stats(self, spike_count):
        self.stats['spikes'].setText(str(spike_count))
        self.stats['time'].setText(f'{self.brain.time:.2f} s')
        self.stats['weight'].setText(f'{np.mean(self.brain.weights):.4f} mV')
        if self.selected is not None:
            self.show_neuron(self.selected)

    def step(self):
        ids = self.brain.step(20*float(self.speed.currentData()))
        self.canvas.record_spikes(ids)
        self.refresh_stats(len(ids))

    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause.setText('Resume')
        else:
            self.timer.start(50)
            self.pause.setText('Pause')

    def pulse(self, pattern):
        self.brain.stimulate(pattern)
        self.detail.setText(f'Injected artificial stimulus {pattern}. This is not yet linguistic input.')

    def train_a(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        ids = np.array([],dtype=int)
        for _ in range(20):
            self.brain.stimulate('A')
            ids = self.brain.step(40)
        self.canvas.record_spikes(ids)
        self.refresh_stats(len(ids))
        self.detail.setText('Presented pattern A 20 times. Compare mean weight and responses; this alone does not prove memory.')
        if was_running:
            self.timer.start(50)

    def show_neuron(self, neuron_id):
        self.selected = int(neuron_id)
        d = self.brain.inspect(neuron_id)
        self.detail.setText(
            f'NEURON {neuron_id}  •  {d["kind"]}  •  Voltage {d["voltage"]:.2f} mV'
            f'  •  Total spikes {d["spikes"]}  •  Inputs {d["incoming"]}  •  Outputs {d["outgoing"]}'
        )

    def save_brain(self):
        filename, _ = QFileDialog.getSaveFileName(self,'Save Pantheon brain',
                           str(Path.cwd()/'memory'/'pantheon_state.pkl'), 'Pantheon state (*.pkl)')
        if not filename: return
        try:
            self.brain.save(filename)
        except Exception as exc:
            QMessageBox.critical(self,'Save failed',str(exc))
        else:
            self.detail.setText(f'Saved neural state to {filename}')

    def load_brain(self):
        filename, _ = QFileDialog.getOpenFileName(self,'Load Pantheon brain',
                           str(Path.cwd()/'memory'), 'Pantheon state (*.pkl)')
        if not filename: return
        running = self.timer.isActive()
        self.timer.stop()
        try:
            self.brain.load(filename)
            self.learn.blockSignals(True)
            self.learn.setChecked(self.brain.learning)
            self.learn.blockSignals(False)
            self.canvas.clear_transients()
            self.refresh_stats(0)
        except Exception as exc:
            QMessageBox.critical(self,'Load failed',str(exc))
        else:
            self.detail.setText(f'Restored Pantheon from {filename}')
        finally:
            if running: self.timer.start(50)
