import xraydb
from PySide6 import QtCore

from expert_pi.gui.main_window import MainWindow
from expert_pi.gui.tools.periodic_table import ElementItem
from expert_pi.stream_processors import edx_processing

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


class XrayController(QtCore.QObject):
    def __init__(self, window: MainWindow, edx_map_processor: edx_processing.EDXMapProcessor) -> None:
        super().__init__()
        self._window = window
        self._xray = window.xray
        self._table = window.xray.periodic_table
        self._edx_map_processor = edx_map_processor

        self.elements: dict[str, ElementItem] = {}
        for atomic_number in range(1, 112):
            lines = {}
            try:
                name = xraydb.atomic_symbol(atomic_number)
                xray_lines: dict[str, xraydb.xray.XrayLine] = xraydb.xray_lines(name)
                for line_name, line in xray_lines.items():
                    if line_name in {"Ka1", "La1", "Ma"} and line.energy <= 30_000:
                        lines[line_name] = line.energy
            except:
                name = missing_names[atomic_number]

            element_item = ElementItem(atomic_number, name, lines, self.element_updated, self.get_auto_color)
            self._table.add_element_button(element_item)

            self.elements[name] = element_item

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self._window = window
        self._xray = window.xray
        self._table = window.xray.periodic_table
        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

        for element_item in self.elements.values():
            self._table.add_element_button(element_item)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {self._xray.add_element_edit.emit_add_element: self.add_element_from_name}

        return signals

    def element_updated(self, element: ElementItem):
        self._table.update_table(element.name)
        self._xray.element_selection.update_elements(element)
        self._window.spectrum_view.update_lines(element)

        self._edx_map_processor.element_change(element)

    def add_element_from_name(self, element_name):
        if element_name in self.elements:
            self.elements[element_name].update_state(selected=True, active=True, color="auto")

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
