from copy import copy

from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel, QFrame, QHBoxLayout, QLineEdit, QFileDialog, QGraphicsRectItem, QSlider, \
    QGraphicsSceneMouseEvent, QTextEdit, QSizePolicy

from PySide6.QtCore import QSize, QPoint, Signal
from PySide6.QtGui import Qt, QGuiApplication
from PySide6 import QtGui
from scipy import ndimage
import numpy as np
import cv2


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def clear_layout(layout):

    while layout.count():
        child = layout.takeAt(0)

        if child.widget():
            child.widget().deleteLater()


def color_filter(frame, c, p1, p2, tx1, tx2, ty1, ty2):
    frame = frame.astype('float64')

    if c == "r":
        a = 0
        b = 1
        c = 2

    elif c == "g":
        a = 1
        b = 0
        c = 2

    else:
        a = 2
        b = 1
        c = 0

    a1 = np.nan_to_num(frame[:, :, a] / (frame[:, :, a] + frame[:, :, b] + frame[:, :, c]), posinf=0, nan=0)
    a2 = frame[:, :, b]
    a3 = frame[:, :, c]

    a1 = (a1 > p1) == (a1 < p2)
    a2 = (a2 > tx1) == (a2 < tx2)
    a3 = (a3 > ty1) == (a3 < ty2)

    result = (a1.astype(np.uint8) * a2.astype(np.uint8) * a3.astype(np.uint8)) * 255

    return result


def roberts_filter(img, v=1, h=1):
    img = img.astype(np.float32)
    img /= 255

    img_v = img.copy()
    img_h = img.copy()

    kernel_v = np.array([[1, 0], [0, -1]])
    kernel_h = np.array([[0, 1], [-1, 0]])

    for c in range(img.shape[2]):
        vertical = ndimage.convolve(img[:, :, c], kernel_v)
        img_v[:, :, c] = vertical

        horizontal = ndimage.convolve(img[:, :, c], kernel_h)
        img_h[:, :, c] = horizontal

        img[:, :, c] = np.sqrt(np.square(vertical) + np.square(horizontal))

    if v == 1 and not h:
        return img_v * 255

    if not v and h == 1:
        return img_h * 255

    return img * 255


def prewitt_filter(img, v=1, h=1):
    img = img.astype(np.float32)
    img /= 255

    img_v = img.copy()
    img_h = img.copy()

    kernel_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_h = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    for c in range(img.shape[2]):
        vertical = ndimage.convolve(img[:, :, c], kernel_v)
        img_v[:, :, c] = vertical

        horizontal = ndimage.convolve(img[:, :, c], kernel_h)
        img_h[:, :, c] = horizontal

        img[:, :, c] = np.sqrt(np.square(vertical) + np.square(horizontal))

    if v == 1 and not h:
        return img_v * 255

    if not v and h == 1:
        return img_h * 255

    return img * 255


def ideal_band_pass_filter(img, l, h):
    img = img.astype(np.float32)

    mask = np.zeros_like(img)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2

    if l:
        mask_l = cv2.circle(copy(mask), (cx, cy), l, (1, 1, 1), -1)

    else:
        mask_l = mask

    if h:
        mask_h = 1 - cv2.circle(copy(mask), (cx, cy), h, (1, 1, 1), -1)

    else:
        mask_h = 1 - mask

    mask = mask_l + mask_h

    for c in range(img.shape[2]):
        dft = cv2.dft(img[:, :, c], flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)

        dft_filtered = mask[:, :, 0:2] * dft_shifted

        idft_shifted = np.fft.ifftshift(dft_filtered)
        idft = cv2.idft(idft_shifted)

        img[:, :, c] = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

    return cv2.convertScaleAbs(img, alpha=255/img.max())


def gaussian_band_pass_filter(img, r_m, l, h):
    img = img.astype(np.float32)

    mask_l = np.exp((-r_m**2) / (2 * l**2))
    mask_h = 1 - np.exp((-r_m**2) / (2 * h**2))
    mask = np.nan_to_num(mask_l) + np.nan_to_num(mask_h)

    mask = np.stack([mask] * 2, axis=-1)

    for c in range(img.shape[2]):
        dft = cv2.dft(img[:, :, c], flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)

        dft_filtered = mask * dft_shifted

        idft_shifted = np.fft.ifftshift(dft_filtered)
        idft = cv2.idft(idft_shifted)

        img[:, :, c] = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

    return cv2.convertScaleAbs(img, alpha=255 / img.max())


def butterworth_low_pass_filter(img, r_m, n, l, h):
    img = img.astype(np.float32)

    mask_l = 1 / (1 + (r_m / l) ** (2 * n))
    mask_h = 1 - 1 / (1 + (r_m / h) ** (2 * n))
    mask = np.nan_to_num(mask_l) + np.nan_to_num(mask_h)

    mask = np.stack([mask] * 2, axis=-1)

    for c in range(img.shape[2]):
        dft = cv2.dft(img[:, :, c], flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)

        dft_filtered = mask * dft_shifted

        idft_shifted = np.fft.ifftshift(dft_filtered)
        idft = cv2.idft(idft_shifted)

        img[:, :, c] = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

    return cv2.convertScaleAbs(img, alpha=255/img.max())


class HSeparator(QFrame):

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class FileSelectBox(QWidget):
    file_selected = Signal(str)

    def __init__(self):
        super().__init__()

        # Structure
        self.body = QHBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)

        # Components
        self.path_le = QLineEdit()
        self.load_btn = QPushButton('...')

        # Functionality
        self.load_btn.clicked.connect(self.get_path)

        # Assembly
        self.body.addWidget(self.path_le)
        self.body.addWidget(self.load_btn)

        self.setLayout(self.body)

    def get_path(self):
        path = QFileDialog.getOpenFileName(parent=self)[0]

        if path:
            self.path_le.setText(path)
            self.file_selected.emit(path)

    def path(self):
        return self.path_le.text()


class CustomWindowFrame(QWidget):

    def __init__(self, title, closable=True, maximizable=True, minimizable=True, movable=True):
        super().__init__()
        self.setFixedHeight(35)

        self.movable = movable

        self.mouse_offset = None
        self.grabbed = False

        # Structure
        self.title_body = QGridLayout()
        self.title_body.setContentsMargins(8, 0, 6, 0)

        self.title_container = QWidget()
        self.title_container.setFixedHeight(20)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 6, 0, 2)

        # Components
        self.title = QLabel(title)
        self.title.setStyleSheet('font-size: 12px')

        self.minimize_btn = QPushButton('–')
        self.minimize_btn.setFixedWidth(20)
        self.minimize_btn.setFlat(True)

        self.maximize_btn = QPushButton('❒')
        self.maximize_btn.setFixedWidth(20)
        self.maximize_btn.setFlat(True)

        self.exit_btn = QPushButton('X')
        self.exit_btn.setFixedWidth(20)
        self.exit_btn.setFlat(True)

        self.separator = HSeparator()

        # Functionality
        self.minimize_btn.clicked.connect(self.minimize)
        self.maximize_btn.clicked.connect(self.maximize)
        self.exit_btn.clicked.connect(self.exit)

        # Assembly
        self.title_body.addWidget(self.title, 0, 0, alignment=Qt.AlignLeft)

        if minimizable:
            self.title_body.addWidget(self.minimize_btn, 0, 2, alignment=Qt.AlignRight)

        if maximizable:
            self.title_body.addWidget(self.maximize_btn, 0, 3, alignment=Qt.AlignRight)

        if closable:
            self.title_body.addWidget(self.exit_btn, 0, 4, alignment=Qt.AlignRight)

        self.title_body.setColumnStretch(1, 1)
        self.title_container.setLayout(self.title_body)

        self.body.addWidget(self.title_container)
        self.body.addWidget(self.separator)

        self.setLayout(self.body)

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mouseDoubleClickEvent(e)
        self.maximize()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        if not self.movable:
            return

        self.mouse_offset = event.position()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(event)

        if not self.movable:
            return

        self.grabbed = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)

        if not self.movable:
            return

        if not self.grabbed and event.position().y() < self.height() and self.mouse_offset.y() < self.height():
            self.grabbed = True

        if self.grabbed:
            x, y = event.globalPosition().toTuple()
            self.parent().move(x - self.mouse_offset.x(), y - self.mouse_offset.y())

    def minimize(self):
        self.parent().showMinimized()

    def maximize(self):

        if self.parent().size() == QGuiApplication.screens()[0].size():
            self.parent().resize(QSize(530, 390))
            self.parent().move(
                QGuiApplication.screens()[0].geometry().center() - self.parent().frameGeometry().center())

        else:
            self.parent().resize(QGuiApplication.screens()[0].size())
            self.parent().move(QPoint(0, 0))

    def exit(self):
        self.parent().hide()


class CustomSlider(QWidget):
    value_changed = Signal()

    def __init__(self, mn, mx, default=None):
        super().__init__()

        # Structure
        self.body = QHBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)

        # Components
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMinimum(mn)
        self.slider.setMaximum(mx)

        if default is not None:
            self.slider.setValue(default)

        self.counter_le = QLineEdit(str(self.slider.value()))
        self.counter_le.setAlignment(Qt.AlignCenter)
        self.counter_le.setFixedWidth(50)

        # Functionality
        self.slider.valueChanged.connect(self.value_changed)
        self.slider.valueChanged.connect(self.update_counter)

        self.counter_le.returnPressed.connect(lambda: self.slider.setValue(int(self.counter_le.text())))

        # Assembly
        self.body.addWidget(self.slider)
        self.body.addWidget(self.counter_le)

        self.setLayout(self.body)

    def update_counter(self):
        self.counter_le.setText(str(self.slider.value()))

    def __getattr__(self, item):
        return getattr(self.slider, item)


class GraphicsItemFrame(QWidget):

    def __init__(self, title):
        super().__init__()
        self.setFixedHeight(35)

        self.mouse_offset = None
        self.grabbed = False

        # Structure
        self.body = QVBoxLayout()
        self.body.setContentsMargins(4, 6, 4, 2)

        # Components
        self.title = QLabel(title)
        self.title.setStyleSheet('font-size: 12px')

        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)

        # Assembly
        self.body.addWidget(self.title, alignment=Qt.AlignCenter)
        self.body.addWidget(self.separator)

        self.setLayout(self.body)


class CustomGraphicsRectItem(QGraphicsRectItem):

    def mouseMoveEvent(self, e: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(e)

        node = self.childItems()[0].widget()
        node.update_connections()
