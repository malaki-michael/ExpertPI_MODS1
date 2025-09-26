from PySide6 import QtWidgets


class Tool(QtWidgets.QGraphicsItemGroup):
    def __init__(self, view):
        super().__init__()
        self.view = view
        self.is_active = False

    def update(self, *args, **kwargs):
        pass

    def hover_enter(self):
        pass

    def hover_leave(self):
        pass

    def drag(self, x, y):
        pass

    def show(self):
        self.is_active = True
        super().show()
        self.update()

    def hide(self):
        self.is_active = False
        super().hide()

    def view_mouse_pressed(self, event, focused_item=None):
        pass

    def view_mouse_moved(self, event, focused_item=None):
        pass

    def view_mouse_released(self, event, focused_item=None):
        pass

    def view_mouse_leaved(self, event, focused_item=None):
        pass
