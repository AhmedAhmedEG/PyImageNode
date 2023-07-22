from PySide6.QtWidgets import QWidget, QGraphicsRectItem, QVBoxLayout, QLabel, QMenu, QGraphicsItem, QGraphicsLineItem, QFormLayout, QFileDialog, QCheckBox, \
    QRadioButton
from Modules.Props import GraphicsItemFrame, FileSelectBox, resize, CustomGraphicsRectItem, CustomSlider, roberts_filter, prewitt_filter, HSeparator, \
    clear_layout, butterworth_low_pass_filter, ideal_band_pass_filter, gaussian_band_pass_filter
from PySide6.QtGui import QImage, QAction, QPixmap, QPen, QColor
from PySide6.QtCore import Signal, QTimer, QLineF, QRect
from PySide6 import QtGui
import cv2
import numpy as np


class ImageNode(QWidget):
    image_changed = Signal()
    delete_triggered = Signal()

    def __init__(self, graphics_item: QGraphicsRectItem, operation='', source=None):
        super().__init__()
        self.original_image = None
        self.processed_image = None
        self.cap = None
        self.first_time = False
        self.setMouseTracking(True)

        self.graphics_item = graphics_item
        self.operation = operation

        if not source:
            self.source = self

        else:
            self.source = source

        self.connections = []

        # Structure
        self.options_body = QFormLayout()
        self.options_container = QWidget()

        self.body = QVBoxLayout()
        self.body.setContentsMargins(2, 2, 2, 2)

        # Components
        self.window_frame = GraphicsItemFrame(operation)
        self.image_viewer_lb = QLabel()

        # Assembly
        self.body.addWidget(self.window_frame)

        if not self.operation:
            self.file_select = FileSelectBox()
            self.file_select.file_selected.connect(self.load_image)
            self.body.addWidget(self.file_select)

        self.body.addWidget(self.image_viewer_lb)

        self.options_container.setLayout(self.options_body)
        self.body.addWidget(self.options_container)

        self.setLayout(self.body)
        QTimer.singleShot(0, self.adjust_parent)

    def contextMenuEvent(self, e: QtGui.QContextMenuEvent) -> None:
        color_filters = ['Grayscale', 'Utsu Binarization', 'Thresholding', 'Adaptive Thresholding']
        linear_filters = ['Average', 'Gaussian']
        non_linear_filters = ['Min', 'Max', 'Median']
        derivation_filters = ['Sobel', 'Roberts', 'Prewitt', 'Laplacian']
        frequency_domain_filters = ['Ideal Band Pass', 'Gaussian Band Pass', 'Butterworth Band Pass']
        edge_detectors = ['Canny']
        enhancement_filters = ['Equalize Histogram', 'Sharpen', 'Brighten', 'Darken']

        menu = QMenu()
        filters_menu = menu.addMenu('Apply Filter')

        color_filters_menu = filters_menu.addMenu('Color Filters')
        for f in color_filters:
            a = color_filters_menu.addAction(f)
            a.setData(f)

        linear_filters_menu = filters_menu.addMenu('Linear Filters')
        for f in linear_filters:
            a = linear_filters_menu.addAction(f)
            a.setData(f)

        non_linear_filters_menu = filters_menu.addMenu('Non-Linear Filters')
        for f in non_linear_filters:
            a = non_linear_filters_menu.addAction(f)
            a.setData(f)

        first_order_filters_menu = filters_menu.addMenu('Derivation Filters')
        for f in derivation_filters:
            a = first_order_filters_menu.addAction(f)
            a.setData(f)

        frequency_domain_filters_menu = filters_menu.addMenu('Frequency Domain Filters')
        for f in frequency_domain_filters:
            a = frequency_domain_filters_menu.addAction(f)
            a.setData(f)

        edge_detector_menu = filters_menu.addMenu('Edge Detectors')
        for d in edge_detectors:
            a = edge_detector_menu.addAction(d)
            a.setData(d)

        enhancement_filters_menu = filters_menu.addMenu('Enhancement Filters')
        for f in enhancement_filters:
            a = enhancement_filters_menu.addAction(f)
            a.setData(f)

        save_action = menu.addAction('Save')
        save_action.setData('Save')

        delete_action = menu.addAction('Delete')
        delete_action.setData('Delete')

        menu.triggered.connect(self.handle_actions)
        menu.exec(e.globalPos())

    def handle_actions(self, a):

        if a.data() == 'Save':
            self.save()

        elif a.data() == 'Delete':
            self.delete()

        else:
            self.add_node(a)

    def adjust_parent(self):
        self.setFixedSize(self.body.sizeHint())

        rect = self.graphics_item.rect()
        rect.setWidth(self.width())
        rect.setHeight(self.height())
        self.graphics_item.setRect(rect)

        self.update_connections()

    def load_image(self, path):

        if '.mp4' in path:
            self.cap = cv2.VideoCapture(path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.original_image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)

        else:
            self.cap = None
            self.original_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        self.processed_image = resize(self.original_image, width=300)

        clear_layout(self.options_body)
        self.first_time = False

        self.run()

        self.image_changed.emit()
        QTimer.singleShot(0, self.adjust_parent)

    def add_node(self, action: QAction):
        pos = self.graphics_item.pos()
        pos.setX(pos.x() + self.width() * 2)

        item = CustomGraphicsRectItem(QRect(0, 0, 1, 1))
        item.setFlag(QGraphicsItem.ItemIsMovable)
        item.setPos(pos)
        item.setZValue(1)

        self.graphics_item.scene().addItem(item)

        node = eval(f'{action.data().replace(" ", "")}Node(item, operation=action.data(), source=self)')
        self.image_changed.connect(node.run)
        self.delete_triggered.connect(node.delete)

        proxy = self.graphics_item.scene().addWidget(node)
        proxy.setParentItem(item)

        self.graphics_item.scene().addItem(proxy)
        self.image_changed.emit()

        QTimer.singleShot(0, lambda: self.add_connection(item, node))

    def add_connection(self, item: QGraphicsRectItem, viewer):
        start = self.graphics_item.pos()
        start.setX(start.x() + self.width())
        start.setY(start.y() + self.height() // 2)

        end = item.pos()
        end.setY(end.y() + item.rect().height() // 2)

        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(3)

        connection = self.graphics_item.scene().addLine(QLineF(start, end), pen)

        viewer.connections.append(('l', connection))
        self.connections.append(['r', connection])

    def update_connections(self):

        if self.connections:

            for connection in self.connections:

                if connection[0] == 'r':
                    start = self.graphics_item.pos()
                    start.setX(start.x() + self.width())
                    start.setY(start.y() + self.height() // 2)

                    item: QGraphicsLineItem = connection[1]

                    line: QLineF = item.line()
                    line.setP1(start)

                    item.setLine(line)

                else:
                    end = self.graphics_item.pos()
                    end.setY(end.y() + self.rect().height() // 2)

                    item: QGraphicsLineItem = connection[1]

                    line: QLineF = item.line()
                    line.setP2(end)

                    item.setLine(line)

    def process_image(self, img):

        if not self.first_time:
            width_sl = CustomSlider(300, self.original_image.shape[0])
            width_sl.value_changed.connect(self.run)

            self.options_body.addRow('Width', width_sl)

            if self.cap:
                frame_sl = CustomSlider(0, self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
                frame_sl.value_changed.connect(self.run)

                self.options_body.addRow('Frame', frame_sl)

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.options_body.itemAt(3).widget().value())
            self.original_image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)

        return resize(self.original_image, width=self.options_body.itemAt(1).widget().value())

    def update_image(self, processed):
        pyside_image = QImage(processed, processed.shape[1], processed.shape[0], processed.strides[0], QImage.Format_RGB888)

        self.processed_image = processed
        self.first_time = True
        self.image_viewer_lb.setPixmap(QPixmap.fromImage(pyside_image))

        self.image_changed.emit()
        QTimer.singleShot(0, self.adjust_parent)

    def run(self):
        self.processed_image = self.source.processed_image
        self.update_image(self.process_image(self.processed_image))

    def save(self):
        path = QFileDialog.getSaveFileName(
            parent=self,
            caption='Save',
            dir=f'Output.png',
            filter='PNG File (*.png)')[0]

        if not path:
            return

        cv2.imwrite(path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

    def delete(self):
        self.delete_triggered.emit()

        for c in self.connections:

            if c[0] == 'r':
                continue

            f = list(filter(lambda i: i[1] == c[1], self.source.connections))[0]

            self.source.connections.remove(f)
            self.graphics_item.scene().removeItem(f[1])

        self.graphics_item.scene().removeItem(self.graphics_item)
        self.deleteLater()


class GrayscaleNode(ImageNode):

    @staticmethod
    def process_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class UtsuBinarizationNode(ImageNode):

    @staticmethod
    def process_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


class ThresholdingNode(ImageNode):

    def process_image(self, img):

        if not self.first_time:
            rbs = [QRadioButton() for _ in range(7)]
            rbs[0].setChecked(True)
            rbs[5].setChecked(True)

            for rb in rbs:
                rb.toggled.connect(lambda s: self.run() if s else None)

            threshold_sl = CustomSlider(0, 255, default=127)
            threshold_sl.value_changed.connect(self.run)

            self.options_body.addRow('Binary', rbs[0])
            self.options_body.addRow('Binary Inverted', rbs[1])
            self.options_body.addRow('Trunc', rbs[2])
            self.options_body.addRow('To Zero', rbs[3])
            self.options_body.addRow('To Zero Inverted', rbs[4])
            self.options_body.addRow('Threshold', threshold_sl)

        if self.options_body.itemAt(1).widget().isChecked():
            tf = cv2.THRESH_BINARY

        elif self.options_body.itemAt(3).widget().isChecked():
            tf = cv2.THRESH_BINARY_INV

        elif self.options_body.itemAt(5).widget().isChecked():
            tf = cv2.THRESH_TRUNC

        elif self.options_body.itemAt(7).widget().isChecked():
            tf = cv2.THRESH_TOZERO

        else:
            tf = cv2.THRESH_TOZERO_INV

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, self.options_body.itemAt(11).widget().value(), 255, tf)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


class AdaptiveThresholdingNode(ImageNode):

    def process_image(self, img):

        if not self.first_time:
            rbs = [QRadioButton() for _ in range(4)]
            rbs[0].setChecked(True)
            rbs[2].setChecked(True)

            for rb in rbs:
                rb.toggled.connect(lambda s: self.run() if s else None)

            method_body = QFormLayout()
            method_body.setContentsMargins(0, 0, 0, 0)

            method_container = QWidget()

            bs_sl = CustomSlider(1, 100, default=11)
            bs_sl.value_changed.connect(self.run)

            offset_sl = CustomSlider(0, 100, default=2)
            offset_sl.value_changed.connect(self.run)

            method_body.addRow('Mean', rbs[2])
            method_body.addRow('Gaussian', rbs[3])

            method_container.setLayout(method_body)

            self.options_body.addRow('Binary', rbs[0])
            self.options_body.addRow('Binary Inverted', rbs[1])
            self.options_body.addRow(HSeparator())
            self.options_body.addRow(method_container)
            self.options_body.addRow(HSeparator())
            self.options_body.addRow('Block Size', bs_sl)
            self.options_body.addRow('Offset', offset_sl)

        if self.options_body.itemAt(1).widget().isChecked():
            tf = cv2.THRESH_BINARY

        else:
            tf = cv2.THRESH_BINARY_INV

        if self.options_body.itemAt(5).widget().layout().itemAt(1).widget().isChecked():
            atf = cv2.ADAPTIVE_THRESH_MEAN_C

        else:
            atf = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, atf, tf, self.options_body.itemAt(8).widget().value(), self.options_body.itemAt(10).widget().value())

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


class AverageNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            width_sl = CustomSlider(1, 10)
            width_sl.value_changed.connect(self.run)

            height_sl = CustomSlider(1, 10)
            height_sl.value_changed.connect(self.run)

            self.options_body.addRow('Filter Width', width_sl)
            self.options_body.addRow('Filter Height', height_sl)

        return cv2.blur(img, (self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value()))


class GaussianNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            width_sl = CustomSlider(1, 10)
            width_sl.value_changed.connect(self.run)

            height_sl = CustomSlider(1, 10)
            height_sl.value_changed.connect(self.run)

            self.options_body.addRow('Filter Width', width_sl)
            self.options_body.addRow('Filter Height', height_sl)

        return cv2.GaussianBlur(img, (self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value()), 0)


class MaxNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            width_sl = CustomSlider(1, 10)
            width_sl.value_changed.connect(self.run)

            height_sl = CustomSlider(1, 10)
            height_sl.value_changed.connect(self.run)

            self.options_body.addRow('Filter Width', width_sl)
            self.options_body.addRow('Filter Height', height_sl)

        kernel = np.ones((self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value()), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)


class MinNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            width_sl = CustomSlider(1, 10)
            width_sl.value_changed.connect(self.run)

            height_sl = CustomSlider(1, 10)
            height_sl.value_changed.connect(self.run)

            self.options_body.addRow('Filter Width', width_sl)
            self.options_body.addRow('Filter Height', height_sl)

        kernel = np.ones((self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value()), np.uint8)
        return cv2.erode(img, kernel, iterations=1)


class MedianNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            size_sl = CustomSlider(1, 10)
            size_sl.value_changed.connect(self.run)

            self.options_body.addRow('Filter Size', size_sl)

        return cv2.medianBlur(img, self.options_body.itemAt(1).widget().value())


class SobelNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            v_cb = QCheckBox()
            v_cb.setChecked(True)
            v_cb.stateChanged.connect(self.run)

            h_cb = QCheckBox()
            h_cb.setChecked(True)
            h_cb.stateChanged.connect(self.run)

            size_sl = CustomSlider(1, 10)
            size_sl.value_changed.connect(self.run)

            self.options_body.addRow('Vertical', v_cb)
            self.options_body.addRow('Horizontal', h_cb)
            self.options_body.addRow('Filter Size', size_sl)

        sobel = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=int(self.options_body.itemAt(1).widget().isChecked()),
                          dy=int(self.options_body.itemAt(3).widget().isChecked()), ksize=self.options_body.itemAt(5).widget().value())

        return cv2.convertScaleAbs(sobel)


class RobertsNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            v_cb = QCheckBox()
            v_cb.setChecked(True)
            v_cb.stateChanged.connect(self.run)

            h_cb = QCheckBox()
            h_cb.setChecked(True)
            h_cb.stateChanged.connect(self.run)

            self.options_body.addRow('Vertical', v_cb)
            self.options_body.addRow('Horizontal', h_cb)

        roberts = roberts_filter(img, v=int(self.options_body.itemAt(1).widget().isChecked()), h=int(self.options_body.itemAt(3).widget().isChecked()))
        return cv2.convertScaleAbs(roberts)


class PrewittNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            v_cb = QCheckBox()
            v_cb.setChecked(True)
            v_cb.stateChanged.connect(self.run)

            h_cb = QCheckBox()
            h_cb.setChecked(True)
            h_cb.stateChanged.connect(self.run)

            self.options_body.addRow('Vertical', v_cb)
            self.options_body.addRow('Horizontal', h_cb)

        prewitt = prewitt_filter(img, v=int(self.options_body.itemAt(1).widget().isChecked()), h=int(self.options_body.itemAt(3).widget().isChecked()))
        return cv2.convertScaleAbs(prewitt)


class LaplacianNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            size_sl = CustomSlider(1, 10)
            size_sl.value_changed.connect(self.run)

            self.options_body.addRow('Filter Size', size_sl)

        laplacian = cv2.Laplacian(src=img, ddepth=cv2.CV_64F, ksize=self.options_body.itemAt(1).widget().value())

        return cv2.convertScaleAbs(laplacian)


class IdealBandPassNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            w, h, _ = img.shape
            mx = int(np.sqrt((w / 2) ** 2 + (h / 2) ** 2))

            lr_sl = CustomSlider(0, mx, default=0)
            lr_sl.value_changed.connect(self.run)

            hr_sl = CustomSlider(0, mx, default=0)
            hr_sl.value_changed.connect(self.run)

            self.options_body.addRow('Low Pass Radius', lr_sl)
            self.options_body.addRow('High Pass Radius', hr_sl)

        return ideal_band_pass_filter(img, self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value())


class GaussianBandPassNode(ImageNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius_matrix = None

    def process_image(self, img):
        self.calc_radius_matrix()

        if not self.first_time:
            w, h, _ = img.shape
            mx = int(np.sqrt((w / 2) ** 2 + (h / 2) ** 2))

            lr_sl = CustomSlider(0, mx, default=0)
            lr_sl.value_changed.connect(self.run)

            hr_sl = CustomSlider(0, mx, default=0)
            hr_sl.value_changed.connect(self.run)

            self.options_body.addRow('Low Pass Radius', lr_sl)
            self.options_body.addRow('High Pass Radius', hr_sl)

        return gaussian_band_pass_filter(img, self.radius_matrix, self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value())

    def calc_radius_matrix(self):

        img = self.source.processed_image
        if self.radius_matrix is None or self.radius_matrix.shape != img.shape:
            w, h, _ = img.shape
            x, y = np.indices([w, h])
            self.radius_matrix = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)


class ButterworthBandPassNode(ImageNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius_matrix = None

    def process_image(self, img):
        self.calc_radius_matrix()

        if not self.first_time:
            w, h, _ = img.shape
            mx = int(np.sqrt((w / 2) ** 2 + (h / 2) ** 2))

            o_sl = CustomSlider(1, 10)
            o_sl.value_changed.connect(self.run)

            lr_sl = CustomSlider(0, mx, default=0)
            lr_sl.value_changed.connect(self.run)

            hr_sl = CustomSlider(0, mx, default=0)
            hr_sl.value_changed.connect(self.run)

            self.options_body.addRow('Order', o_sl)
            self.options_body.addRow('Low Pass Radius', lr_sl)
            self.options_body.addRow('High Pass Radius', hr_sl)

        return butterworth_low_pass_filter(img, self.radius_matrix, self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value(),
                                           self.options_body.itemAt(5).widget().value())

    def calc_radius_matrix(self):

        img = self.source.processed_image
        if self.radius_matrix is None or self.radius_matrix.shape != img.shape:
            w, h, _ = img.shape
            x, y = np.indices([w, h])
            self.radius_matrix = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)


class CannyNode(ImageNode):

    def process_image(self, img):
        if not self.first_time:
            low_sl = CustomSlider(0, 255)
            low_sl.value_changed.connect(self.run)

            high_sl = CustomSlider(0, 255)
            high_sl.value_changed.connect(self.run)

            size_sl = CustomSlider(3, 7)
            size_sl.value_changed.connect(self.run)

            self.options_body.addRow('Low Threshold', low_sl)
            self.options_body.addRow('High Threshold', high_sl)
            self.options_body.addRow('Sobel Filter Size', size_sl)

        canny = cv2.Canny(img, self.options_body.itemAt(1).widget().value(), self.options_body.itemAt(3).widget().value(),
                          apertureSize=self.options_body.itemAt(5).widget().value())

        return cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)


class EqualizeHistogramNode(ImageNode):

    @staticmethod
    def process_image(img):
        for c in range(img.shape[2]):
            img[:, :, c] = cv2.equalizeHist(img[:, :, c])

        return img


class BrightenNode(ImageNode):

    def process_image(self, img):

        if not self.first_time:
            r_sl = CustomSlider(0, 100)
            r_sl.value_changed.connect(self.run)

            self.options_body.addRow('Brightness', r_sl)

        inv_gamma = 1 - self.options_body.itemAt(1).widget().value() * 0.008
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(img, table)


class DarkenNode(ImageNode):

    def process_image(self, img):

        if not self.first_time:
            r_sl = CustomSlider(0, 100)
            r_sl.value_changed.connect(self.run)

            self.options_body.addRow('Darkness', r_sl)

        inv_gamma = 1 + self.options_body.itemAt(1).widget().value() * 0.05
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(img, table)


class SharpenNode(ImageNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius_matrix = None

    def process_image(self, img):
        self.calc_radius_matrix()

        if not self.first_time:
            rbs = [QRadioButton() for _ in range(3)]
            rbs[0].setChecked(True)

            for rb in rbs:
                rb.toggled.connect(lambda s: self.run() if s else None)

            w, h, _ = img.shape
            mx = int(np.sqrt((w / 2) ** 2 + (h / 2) ** 2))

            r_sl = CustomSlider(0, mx, default=10)
            r_sl.value_changed.connect(self.run)

            o_sl = CustomSlider(1, 10)
            o_sl.value_changed.connect(self.run)

            w_sl = CustomSlider(0, 100, default=10)
            w_sl.value_changed.connect(self.run)

            self.options_body.addRow('Ideal Filter', rbs[0])
            self.options_body.addRow('Gaussian Filter', rbs[1])
            self.options_body.addRow('Butterworth Filter', rbs[2])
            self.options_body.addRow('Order', o_sl)
            self.options_body.addRow('High Pass Radius', r_sl)
            self.options_body.addRow('Sharpness', w_sl)

        if self.options_body.itemAt(5).widget().isChecked():
            self.options_body.itemAt(6).widget().setVisible(True)
            self.options_body.itemAt(7).widget().setVisible(True)

        else:
            self.options_body.itemAt(6).widget().setVisible(False)
            self.options_body.itemAt(7).widget().setVisible(False)

        if self.options_body.itemAt(1).widget().isChecked():
            edges = ideal_band_pass_filter(img, 0, self.options_body.itemAt(9).widget().value())

        elif self.options_body.itemAt(3).widget().isChecked():
            edges = gaussian_band_pass_filter(img, self.radius_matrix, 0, self.options_body.itemAt(9).widget().value())

        else:
            edges = butterworth_low_pass_filter(img, self.radius_matrix, self.options_body.itemAt(7).widget().value(), 0,
                                                self.options_body.itemAt(9).widget().value())

        return cv2.addWeighted(img, 1, edges, self.options_body.itemAt(11).widget().value() / 100, 0)

    def calc_radius_matrix(self):

        img = self.source.processed_image
        if self.radius_matrix is None or self.radius_matrix.shape != img.shape:
            w, h, _ = img.shape
            x, y = np.indices([w, h])
            self.radius_matrix = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
