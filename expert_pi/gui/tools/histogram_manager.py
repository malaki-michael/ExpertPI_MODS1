from expert_pi.gui.tools import base


class HistogramManager(base.Tool):
    def __init__(self, view):
        super().__init__(view)
        self.histogram = None

    def show(self):
        self.is_active = True
        if self.histogram is not None:
            self.histogram.show()

    def hide(self):
        self.is_active = False
        if self.histogram is not None:
            self.histogram.hide()
