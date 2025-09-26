from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.gui.main_window import MainWindow


class DiffractionsToolsController:
    def __init__(self, window: MainWindow, states: StatesHolder) -> None:
        self._window = window
        self._states = states
        self._tools = window.diffraction_tools

        self.diffraction_generator_first_open = True

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self._window = window
        self._tools = window.diffraction_tools
        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self._tools.open_diffraction_generator.clicked: self.show_diffraction_generator,
            # self._tools.image_type.clicked: self.change_image_type,
            self._tools.visualisation.clicked: self.select_visualisation,
            self._tools.selectors.clicked: self.change_selectors,
            self._tools.tools_expansions["mask_selector"].mask_type.clicked: self.masked_changed,
            self._tools.tools_expansions["mask_selector"].segments.set_signal: self.masked_changed,
        }
        return signals

    def change_selectors(self, name):
        for name in self._tools.selectors.options:
            if self._tools.selectors.selected is None or name not in self._tools.selectors.selected:
                self._window.diffraction_view.tools[name].hide()
            else:
                self._window.diffraction_view.tools[name].show()
        self._window.diffraction_view.update()

    def masked_changed(self, *args, **kwargs):
        if not self._window.diffraction_view.tools["mask_selector"].is_active:
            return

        mask_selector = self._window.diffraction_view.tools["mask_selector"]

        if self._tools.tools_expansions["mask_selector"].mask_type.selected == "angular":
            if mask_selector.radii[0] == 0:
                mask_selector.radii[0] = mask_selector.radii[1] / 2
        else:
            mask_selector.radii[0] = 0

        segments = self._tools.tools_expansions["mask_selector"].segments.value()
        mask_selector.set_segments(segments)  # will emit change every time

    def select_visualisation(self, name):
        if self._tools.visualisation.selected is not None and name in self._tools.visualisation.selected:
            self._window.diffraction_view.tools[name].show()
        else:
            self._window.diffraction_view.tools[name].hide()
        self._window.diffraction_view.update()

    def show_diffraction_generator(self):
        self._tools.diffraction_generator.reload()
        self._tools.diffraction_generator.show()
        if self.diffraction_generator_first_open:
            self._tools.diffraction_generator.setGeometry(
                self._window.geometry().x() + 200, self._window.geometry().y() + 70, 800, 600
            )
        self.diffraction_generator_first_open = False
