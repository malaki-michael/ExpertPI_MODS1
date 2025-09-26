import os

from PySide6 import QtWidgets

from expert_pi.app.states import preserved_state_saver


def image_file_save_dialog(parent_widget, data_folder: str):
    last_folder = preserved_state_saver.actual_state["last_save_directory"]

    name, *_ = QtWidgets.QFileDialog.getSaveFileName(
        parent_widget, "Save File", last_folder + "untitled.tiff", "tiff file (*.tiff);;png file (*.png)"
    )
    if not name:
        return None

    preserved_state_saver.actual_state["last_save_directory"] = os.path.dirname(name) + "/"
    preserved_state_saver.save(data_folder)

    return name
