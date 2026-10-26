"""Read-only organic visualization of actual neurons and synapses."""
import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget


class NeuralCanvas(QWidget):
    neuron_selected = Signal(int)

    def __init__(self, brain):
        super().__init__()
        self.brain = brain
        rng = np.random.default_rng(17)
        theta = rng.uniform(0, 2*np.pi, len(brain.neurons))
        radius = np.sqrt(rng.uniform(0, 1, len(brain.neurons)))
        self.positions = np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))
        self.glow = np.zeros(len(brain.neurons))
        self.sources = np.zeros(len(brain.neurons))
        self.selected = None
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.drag_origin = None
        self.dragged = False
        self.setMinimumSize(440, 330)
        self.setCursor(Qt.OpenHandCursor)

    def points(self):
        scale = min(self.width(), self.height())*0.47*self.zoom
        center = QPointF(self.width()/2, self.height()/2) + self.pan
        return [QPointF(center.x()+x*scale, center.y()+y*scale) for x, y in self.positions]

    def record_spikes(self, ids):
        self.glow *= 0.82
        self.sources *= 0.70
        if len(ids):
            self.glow[ids] = 1
            self.sources[ids] = 1
        self.update()

    def clear_transients(self):
        self.glow[:] = 0
        self.sources[:] = 0
        self.update()

    def reset_view(self):
        self.zoom = 1
        self.pan = QPointF(0, 0)
        self.update()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        new_zoom = float(np.clip(self.zoom*factor, .35, 8))
        ratio = new_zoom/self.zoom
        center = QPointF(self.width()/2, self.height()/2)
        pointer = event.position()
        self.pan = pointer-center-(pointer-center-self.pan)*ratio
        self.zoom = new_zoom
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_origin = event.position()
            self.dragged = False
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag_origin is not None:
            delta = event.position()-self.drag_origin
            if abs(delta.x())+abs(delta.y()) > 2:
                self.dragged = True
            self.pan += delta
            self.drag_origin = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.OpenHandCursor)
            self.drag_origin = None
            if not self.dragged:
                points = self.points()
                distances = [(p-event.position()).manhattanLength() for p in points]
                index = int(np.argmin(distances))
                if distances[index] < 13:
                    self.selected = index
                    self.neuron_selected.emit(index)
                    self.update()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#080f1f'))
        pts = self.points()
        weights = self.brain.weights
        for k, (i, j, excitatory) in enumerate(self.brain.edges):
            pulse = self.sources[i]
            base = QColor('#609fc5') if excitatory else QColor('#d49b65')
            selected = self.selected is not None and (i == self.selected or j == self.selected)
            weight = weights[k] if excitatory else 1.5
            base.setAlpha(int(min(220, 12 + pulse*100 + (40 if selected else 0) + (weight*6 if excitatory else 0))))
            painter.setPen(QPen(base, 0.55 + pulse*1.15 + (0.9 if selected else 0)))
            painter.drawLine(pts[i], pts[j])
        painter.setPen(Qt.NoPen)
        for i, point in enumerate(pts):
            pulse = self.glow[i]
            base = QColor('#61a7d0') if i < 205 else QColor('#dd9d66')
            if pulse > 0.04:
                halo = QColor(base)
                halo.setAlpha(int(75*pulse))
                painter.setBrush(QBrush(halo))
                painter.drawEllipse(point, 3.5+pulse*5, 3.5+pulse*5)
            painter.setBrush(QBrush(QColor('#f0fbff') if pulse > .4 else base))
            r = 1.7+1.8*pulse
            painter.drawEllipse(point, r, r)
            if self.selected == i:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor('#ffffff'), 1.5))
                painter.drawEllipse(point, 8, 8)
                painter.setPen(Qt.NoPen)
        painter.end()
