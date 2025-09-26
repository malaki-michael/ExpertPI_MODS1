from collections.abc import Callable

import xraydb
from PySide6 import QtCore

edx_coloring_map = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c"]

missing_names = {
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cp",
    113: "Uut",
    114: "Uuq",
    115: "Uup",
    116: "Uuh",
    117: "Uus",
    118: "Uuo",
    119: "Uue",
}


class ElementItem:
    def __init__(self, atomic_number, update_signal, get_auto_color):
        self.atomic_number = atomic_number
        self._update_signal = update_signal
        self.get_auto_color = get_auto_color

        try:
            self.name = xraydb.atomic_symbol(atomic_number)
            lines = xraydb.xray_lines(self.name)
        except:
            self.name = missing_names[atomic_number]
            lines = {}

        self.lines = {}
        for line_name, line in lines.items():
            if line_name in {"Ka1", "La1", "Ma"} and line.energy < 30_000:
                self.lines[line_name] = line

        self.selected = False
        self.active = False
        self.hover = False
        self.color = ""

    def update_state(self, selected: bool = None, active: bool = None, hover: bool = None, color: str = None):
        if selected is not None:
            self.selected = selected
        if active is not None:
            self.active = active
        if hover is not None:
            self.hover = hover
        if color is not None:
            if color == "auto":
                color = self.get_auto_color()
            self.color = color

        self._update_signal.emit(self.name)


class PeriodicTableModel(QtCore.QObject):
    _model_updated = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.elements = {}
        self.observers = {}

    def register_observer(self, name: str, update_function: Callable[[str], None]):
        if name in self.observers:
            fn = self.observers.pop(name)
            self._model_updated.disconnect(fn)

        self.observers[name] = update_function
        self._model_updated.connect(update_function)

    def add_element(self, atomic_number):
        element = ElementItem(atomic_number, self._model_updated, self.get_auto_color)
        self.elements[element.name] = element

        return element

    def get_selected_items(self):
        return [self.elements[name] for name in self.elements if self.elements[name].selected]

    def get_auto_color(self):
        selected_elements = self.get_selected_items()
        selected_colors = [element.color for element in selected_elements]
        for color in edx_coloring_map:
            if color not in selected_colors:
                break
        else:
            color = "#ffffff"

        return color
