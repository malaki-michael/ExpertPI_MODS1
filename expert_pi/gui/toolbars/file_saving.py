from PySide6 import QtCore, QtWidgets

from expert_pi.gui.elements import buttons
from expert_pi.gui.style import images_dir


class FileSaving(QtWidgets.QWidget):
    save_signal = QtCore.Signal(str, bool)
    new_signal = QtCore.Signal()
    opened_change_signal = QtCore.Signal(bool)

    status_signal = QtCore.Signal(str, bool, bool)  # incoming signal (filename, open, modified)

    def __init__(self):
        super().__init__()

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.opened_checkbox = QtWidgets.QCheckBox("")
        self.opened_checkbox.setToolTip("unmark to save and close h5 file")

        self.opened_checkbox.clicked.connect(self.opened_checkbox_clicked)

        self.filename = QtWidgets.QLineEdit("")
        self.filename.setPlaceholderText("In memory")
        self.filename.setEnabled(False)
        self.filename.setFixedWidth(500)

        self.file = None

        self.layout().addWidget(self.opened_checkbox)
        self.layout().addWidget(self.filename)

        self.save_button = buttons.ToolbarPushButton("", icon=images_dir + "tools_icons/save.svg", tooltip="save ")
        self.new_button = buttons.ToolbarPushButton(
            "", icon=images_dir + "tools_icons/new.svg", tooltip="open new file"
        )

        self.save_button.setProperty("class", "toolbarButton big")
        self.new_button.setProperty("class", "toolbarButton big")

        self.layout().addWidget(self.save_button)
        self.layout().addWidget(self.new_button)

        self.save_button.clicked.connect(self.save)
        self.new_button.clicked.connect(self.new)

        self.status_signal.connect(self.status_changed)

    def status_changed(self, filename, open, modified):
        self.filename.setText(filename)
        self.opened_checkbox.setChecked(open)

        self.save_button.setProperty("error", modified)
        self.save_button.setStyleSheet(self.save_button.styleSheet())  # need to redraw the selected property
        if modified:
            self.filename.setStyleSheet("color:red")
        else:
            self.filename.setStyleSheet("")

    def new(self):
        self.new_signal.emit()  # the controler must send back the new filename via modified signal

    def save(self, close=False):
        self.save_button.setProperty("error", False)
        self.filename.setStyleSheet("")
        self.save_button.setStyleSheet(self.save_button.styleSheet())  # need to redraw the selected property
        self.save_signal.emit(self.filename.text(), close)

    def opened_checkbox_clicked(self):
        self.opened_change_signal.emit(self.opened_checkbox.isChecked())

    def hide(self):
        super().hide()

        for item in [self.opened_checkbox, self.filename, self.save_button, self.new_button]:
            item.setVisible(False)

    def show(self, filename=""):
        super().show()
        self.filename.setText(filename)
        for item in [self.opened_checkbox, self.filename, self.save_button, self.new_button]:
            item.setVisible(True)
