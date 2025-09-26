from PySide6 import QtCore

from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.gui.main_window import MainWindow


class SurveyController(QtCore.QObject):
    def __init__(self, window: MainWindow, states: StatesHolder) -> None:
        super().__init__()
        self.window = window
        self.states = states

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self.window = window
        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {self.window.survey_view.show_signal: self.show}
        return signals

    def show(self):
        self.states.interaction_mode = StatesHolder.InteractionMode.survey
        self.window.tool_bar.file_saving.hide()
