from PySide6 import QtWidgets


class ToolbarManager:
    def __init__(self, parent, panel_size, type: str = "left"):
        self.parent = parent
        self.type = type
        self.panel_size = panel_size
        self.toolbars = {}

        self.visible_toolbars = []  # they will be showed in order according to the this list

        self.scrollbar = QtWidgets.QScrollBar()
        self.scrollbar.setParent(self.parent)
        self.scrollbar.hide()
        self.scrollbar_width = 12
        self.scrollbar.valueChanged.connect(self.slider_changed)

        self.visible = True

    def add_toolbar(self, toolbar, set_parent=True):
        name = toolbar.name
        if name in self.toolbars:
            raise Exception(name + " already in toolbar")
        if set_parent:
            toolbar.setParent(self.parent)
        self.toolbars[name] = toolbar
        self.visible_toolbars.append(name)
        toolbar.toolbar_manager = self
        return toolbar

    def slider_changed(self, value):
        self.parent_resized()

    def parent_resized(self, size=None):
        if size is None:
            size = self.parent.size()
        total_size = 0
        add = 5

        for name, toolbar in self.toolbars.items():
            if name not in self.visible_toolbars:
                toolbar.hide()

        for name in self.visible_toolbars:
            total_size += add + self.toolbars[name].sizeHint().height()

        scroll_bar_width = 0
        vertical_position = 0

        if total_size > size.height():
            self.scrollbar.show()
            self.scrollbar.setMaximum(total_size - size.height())
            self.scrollbar.setPageStep(size.height())
            self.scrollbar.setFixedSize(self.scrollbar_width, size.height())
            if self.type == "left":
                self.scrollbar.move(0, 0)
            else:
                self.scrollbar.move(size.width() - self.scrollbar_width, 0)
            scroll_bar_width = self.scrollbar_width
            vertical_position = -self.scrollbar.value()
        else:
            self.scrollbar.hide()

        for name in self.visible_toolbars:
            toolbar = self.toolbars[name]
            toolbar.show()
            toolbar.slider_width = scroll_bar_width
            if toolbar.expanded:
                toolbar.setFixedWidth(self.panel_size * 2 - scroll_bar_width)
                if self.type == "right":
                    toolbar.move(size.width() - self.panel_size * 2, vertical_position)
                else:
                    toolbar.move(scroll_bar_width, vertical_position)
            else:
                toolbar.setFixedWidth(self.panel_size - scroll_bar_width)
                if self.type == "right":
                    toolbar.move(size.width() - self.panel_size, vertical_position)
                else:
                    toolbar.move(scroll_bar_width, vertical_position)

            vertical_position += toolbar.sizeHint().height() + add

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False
