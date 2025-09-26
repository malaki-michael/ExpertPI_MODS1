from expert_pi.gui import main_window
from expert_pi.gui.main_window import MainWindow


class StemToolsController:
    def __init__(self, window: main_window.MainWindow) -> None:
        self._window = window
        self._tools = self._window.stem_tools

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self._window = window
        self._tools = self._window.stem_tools
        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self._tools.visualisation.clicked: self.select_visualisation,
            self._tools.selectors.clicked: self.change_selectors,
        }

        return signals

    def change_selectors(self, name):
        for name in self._tools.selectors.options:
            if self._tools.selectors.selected is None or name not in self._tools.selectors.selected:
                self._window.image_view.tools[name].hide()
            else:
                self._window.image_view.tools[name].show()
        self._window.image_view.update()

    def select_visualisation(self, name):
        if self._tools.visualisation.selected is not None and name in self._tools.visualisation.selected:
            self._window.image_view.tools[name].show()
        else:
            self._window.image_view.tools[name].hide()
        self._window.image_view.update()
