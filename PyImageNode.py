from PySide6.QtWidgets import QApplication, QWidget, QGraphicsScene, QVBoxLayout, QGraphicsView, QGraphicsItem, QMenu, QToolBar
from Modules.Props import CustomWindowFrame, CustomGraphicsRectItem
from PySide6.QtGui import QPainter, Qt, QPalette, QColor
from PySide6.QtCore import QSize, QRect, QRectF
from Modules.ImageNode import ImageNode
from PySide6 import QtGui


class CustomGraphicsView(QGraphicsView):

    def __init__(self, *args):
        super().__init__(*args)
        self.scale = 1
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setAcceptDrops(True)

    def wheelEvent(self, event):

        if event.angleDelta().y() > 0:
            self.scale = 1.1
            self.setTransform(self.transform().scale(self.scale, self.scale))

        else:
            self.scale = 0.9
            self.setTransform(self.transform().scale(self.scale, self.scale))

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:

        if event.mimeData().hasUrls():
            event.accept()

        else:
            event.ignore()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        self.add_image_node(self.mapToScene(e.pos()).toTuple(), path=e.mimeData().urls()[0].toLocalFile())

    def contextMenuEvent(self, e: QtGui.QContextMenuEvent) -> None:

        item = self.itemAt(e.pos())
        if item:
            super().contextMenuEvent(e)
            return

        menu = QMenu()

        add_action = menu.addAction('Add')
        add_action.triggered.connect(lambda: self.add_image_node(self.mapToScene(e.pos()).toTuple()))

        menu.exec(e.globalPos())

    def add_image_node(self, pos, path=''):
        item = CustomGraphicsRectItem(QRectF(0, 0, 1, 1))
        item.setFlag(QGraphicsItem.ItemIsMovable)
        item.setPos(*pos)
        item.setZValue(1)

        self.scene().addItem(item)

        node = ImageNode(item)
        if path:
            node.load_image(path)
            node.file_select.path_le.setText(path)

        proxy = self.scene().addWidget(node)
        proxy.setParentItem(item)

        self.scene().addItem(proxy)


class PyImageGUI(QWidget):

    def __init__(self):
        super(PyImageGUI, self).__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)

        # Structure
        self.boby = QVBoxLayout()
        self.boby.setContentsMargins(0, 0, 0, 0)
        self.boby.setSpacing(0)

        # Components
        self.window_frame = CustomWindowFrame(title='PyImageGUI')

        self.scene = QGraphicsScene()
        self.scene.setSceneRect(QRect(0, 0, 50000, 50000))

        self.view = CustomGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        # Assembly
        self.boby.addWidget(self.window_frame)
        self.boby.addWidget(self.view)

        self.setLayout(self.boby)


if '__main__' in __name__:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#2a2a2a"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#353535"))

    app = QApplication([])
    app.setStyle('Fusion')
    app.setPalette(palette)
    app.setStyleSheet('''QWidget {color: #ffffff}
                         QWidget:!enabled {color: #808080}''')

    window = PyImageGUI()
    window.resize(QSize(1280, 720))
    window.show()

    app.exec()
