from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PyQt5 import uic
from PIL import Image, ImageDraw, ImagePalette, ImageFont

from collections.abc import Iterable
import sys
import os
import itertools
import random
import re


IMAGE_ENCRYPT_FILENAME = "image_encrypt.png"
IMAGE_DECRYPT_FILENAME = "image_decrypt.png"
PRIMARY_IMAGE_FILENAME = f"primary_image.png"
COMPONENT_INDEX = {'t': 3, 'b': 2, 'g': 1, 'r': 0}
EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'gif']


def except_exceptions(clas_, exception, callback):
    sys.__excepthook__(clas_, exception, callback)


class StegSolver(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("StegoSolver.ui", self)
        self.file_decrypt_chosen = None
        self.file_decrypt_image = None
        self.file_encrypt_chosen = None
        self.file_encrypt_image = None

        self.w, self.h = None, None
        self.text = self.x_pos = self.y_pos = self.size = None

        self.open_image_btn.clicked.connect(self.open_decrypt_image)
        self.open_encrypt_image_btn.clicked.connect(self.open_encrypt_image)

        self.pixels = []

        for colour in ('Red', 'Green', 'Blue', 'Alpha'):
            for number in range(1, 9):
                self.encryption_formats_box.addItem(f"{colour} format {number}")

        self.red_format_1.clicked.connect(lambda: self.format_image_by_component(1, 'r'))
        self.red_format_2.clicked.connect(lambda: self.format_image_by_component(2, 'r'))
        self.red_format_3.clicked.connect(lambda: self.format_image_by_component(3, 'r'))
        self.red_format_4.clicked.connect(lambda: self.format_image_by_component(4, 'r'))
        self.red_format_5.clicked.connect(lambda: self.format_image_by_component(5, 'r'))
        self.red_format_6.clicked.connect(lambda: self.format_image_by_component(6, 'r'))
        self.red_format_7.clicked.connect(lambda: self.format_image_by_component(7, 'r'))
        self.red_format_8.clicked.connect(lambda: self.format_image_by_component(8, 'r'))

        self.green_format_1.clicked.connect(lambda: self.format_image_by_component(1, 'g'))
        self.green_format_2.clicked.connect(lambda: self.format_image_by_component(2, 'g'))
        self.green_format_3.clicked.connect(lambda: self.format_image_by_component(3, 'g'))
        self.green_format_4.clicked.connect(lambda: self.format_image_by_component(4, 'g'))
        self.green_format_5.clicked.connect(lambda: self.format_image_by_component(5, 'g'))
        self.green_format_6.clicked.connect(lambda: self.format_image_by_component(6, 'g'))
        self.green_format_7.clicked.connect(lambda: self.format_image_by_component(7, 'g'))
        self.green_format_8.clicked.connect(lambda: self.format_image_by_component(8, 'g'))

        self.blue_format_1.clicked.connect(lambda: self.format_image_by_component(1, 'b'))
        self.blue_format_2.clicked.connect(lambda: self.format_image_by_component(2, 'b'))
        self.blue_format_3.clicked.connect(lambda: self.format_image_by_component(3, 'b'))
        self.blue_format_4.clicked.connect(lambda: self.format_image_by_component(4, 'b'))
        self.blue_format_5.clicked.connect(lambda: self.format_image_by_component(5, 'b'))
        self.blue_format_6.clicked.connect(lambda: self.format_image_by_component(6, 'b'))
        self.blue_format_7.clicked.connect(lambda: self.format_image_by_component(7, 'b'))
        self.blue_format_8.clicked.connect(lambda: self.format_image_by_component(8, 'b'))

        self.alpha_format_1.clicked.connect(lambda: self.format_image_by_component(1, 't'))
        self.alpha_format_2.clicked.connect(lambda: self.format_image_by_component(2, 't'))
        self.alpha_format_3.clicked.connect(lambda: self.format_image_by_component(3, 't'))
        self.alpha_format_4.clicked.connect(lambda: self.format_image_by_component(4, 't'))
        self.alpha_format_5.clicked.connect(lambda: self.format_image_by_component(5, 't'))
        self.alpha_format_6.clicked.connect(lambda: self.format_image_by_component(6, 't'))
        self.alpha_format_7.clicked.connect(lambda: self.format_image_by_component(7, 't'))
        self.alpha_format_8.clicked.connect(lambda: self.format_image_by_component(8, 't'))

        self.full_red_btn.clicked.connect(lambda: self.fill_one_colour('r'))
        self.full_green_btn.clicked.connect(lambda: self.fill_one_colour('g'))
        self.full_blue_btn.clicked.connect(lambda: self.fill_one_colour('b'))

        self.invert_btn.clicked.connect(self.invert)
        self.grey_btn.clicked.connect(self.make_grey)
        self.bw_btn.clicked.connect(self.make_bw)
        self.random_btn.clicked.connect(self.make_randomized)

        self.reflect_rl_btn.clicked.connect(lambda: self.reflect('lr'))
        self.reflect_tb_btn.clicked.connect(lambda: self.reflect('tb'))

        self.reset_btn.clicked.connect(self.reset)

        self.decrypt_page_btn.clicked.connect(lambda: self.main_frame.setCurrentIndex(0))
        self.encrypt_page_btn.clicked.connect(lambda: self.main_frame.setCurrentIndex(1))

        self.size_le.textChanged.connect(self.validate_digit_input)
        self.x_le.textChanged.connect(self.validate_digit_input)
        self.y_le.textChanged.connect(self.validate_digit_input)
        self.preview_btn.clicked.connect(self.preview)
        self.save_encrypt_image_btn.clicked.connect(self.save_encrypted_image)
        self.save_decrypted_image_btn.clicked.connect(self.save_decrypted_image)

    def preview(self) -> None:
        """Preview of an image on a encryption page"""
        if not self.file_encrypt_chosen:
            self.make_dialog(f"File is not selected")
            return

        if not self.text_le.text():
            self.make_dialog(f"Text is not specified")
            return

        self.text = self.text_le.text()
        self.x_pos = self.w // 2 if not self.x_le.text() else int(self.x_le.text())
        self.y_pos = self.h // 2 if not self.y_le.text() else int(self.y_le.text())
        self.size = 50 if not self.size_le.text() else int(self.size_le.text())

        image = Image.open(self.file_encrypt_chosen)
        drawer = ImageDraw.Draw(image)
        fnt = ImageFont.truetype(font='consolas.ttf', size=self.size)

        try:
            drawer.text((self.x_pos, self.y_pos), self.text, font=fnt, fill=(255, 255, 255))
        except ValueError:
            self.make_dialog(f"Unavailable option for current image mode: {image.mode}")
            return

        self.file_encrypt_image = IMAGE_ENCRYPT_FILENAME
        image.save(self.file_encrypt_image)
        self.image_lbl_2.setPixmap(QPixmap(self.file_encrypt_image).scaled(self.image_lbl_2.width(), self.image_lbl_2.height(), Qt.KeepAspectRatio))

    def validate_digit_input(self) -> None:
        """Validates the input"""
        self.sender().setText(re.sub(r'\D', "", self.sender().text()))

    @staticmethod
    def _generate_format_values(format_type: int) -> Iterable[int]:
        """Generates a tuple of values according to the format"""
        return tuple(itertools.chain.from_iterable(tuple(range(i, i + (2 ** (format_type - 1)))) for i in range(0, 256, 2 ** format_type)))

    @staticmethod
    def _get_image_data(filename: str, mode: (str, None) = 'RGBA') -> Iterable:
        if mode is None:
            image = Image.open(filename)
        else:
            image = Image.open(filename).convert(mode)
        x, y = image.size
        pixels = image.load()
        return image, pixels, x, y

    @staticmethod
    def _get_distinct_colour(iterable: Iterable) -> Iterable:
        for i in range(256):
            for j in range(256):
                for k in range(256):
                    if (i, j, k) not in iterable:
                        return i, j, k
        return (0, ) * 3  # if no colour found return black

    @staticmethod
    def _find_nearest_value(array: Iterable, value: int, format_type: int) -> int:
        delta = 256
        res = value

        for i in range(value - 2 ** (int(format_type) - 1) - 1, value + 2 ** (int(format_type) - 1) + 1):
            if i in array:
                if (temp := abs(i - value)) <= delta:
                    res = i
                    delta = temp
        return res

    def save_decrypted_image(self) -> None:
        image_file = QFileDialog.getSaveFileName(self, "Save image as...", f"C:\\encrypted_image.png",
                                                           'Image (*.png);;Image (*.jpg);;'
                                                           'Image (*.jpeg);;Image (*.bmp);;')[0]
        if not image_file:
            return

        Image.open(self.file_decrypt_image).save(image_file)

    def save_encrypted_image(self) -> None:
        if not self.file_encrypt_chosen:
            self.make_dialog(f"File is not selected")
            return

        if not self.text_le.text():
            self.make_dialog(f"Text is not specified")
            return

        self.preview()

        self.label_wait.setText(f"Please wait. It can take up to 60 seconds.")

        ordinary_image = Image.open(self.file_encrypt_chosen).convert('RGBA')
        ordinary_pixels = ordinary_image.load()
        image_colours = {ordinary_pixels[i, j][:3] for i in range(self.w) for j in range(self.h)}
        text_colour = self._get_distinct_colour(sorted(image_colours))

        drawer = ImageDraw.Draw(ordinary_image)
        fnt = ImageFont.truetype(font='consolas.ttf', size=self.size)
        drawer.text((self.x_pos, self.y_pos), self.text, font=fnt, fill=text_colour)
        ordinary_image.save(PRIMARY_IMAGE_FILENAME)

        ordinary_image = Image.open(self.file_encrypt_chosen)
        ordinary_pixels = ordinary_image.load()

        format_colour, *_, format_type = self.encryption_formats_box.currentText().lower().split(' ')

        primary_image = Image.open(PRIMARY_IMAGE_FILENAME).convert('RGBA')
        primary_pixels = primary_image.load()  #

        list_of_values_for_text = list(self._generate_format_values(int(format_type)))
        list_of_values_for_bg = list(set(range(256)).difference(set(list_of_values_for_text)))

        for i in range(self.w):
            for j in range(self.h):
                r, g, b, t = primary_pixels[i, j]
                if (r, g, b) == text_colour:  # If it's text
                    if format_colour == 'red':
                        ordinary_pixels[i, j] = self._find_nearest_value(list_of_values_for_text, ordinary_pixels[i, j][0], format_type), ordinary_pixels[i, j][1], ordinary_pixels[i, j][2], ordinary_pixels[i, j][3]
                    elif format_colour == 'green':
                        ordinary_pixels[i, j] = ordinary_pixels[i, j][0], self._find_nearest_value(list_of_values_for_text, ordinary_pixels[i, j][1], format_type), ordinary_pixels[i, j][2], ordinary_pixels[i, j][3]
                    elif format_colour == 'blue':
                        ordinary_pixels[i, j] = ordinary_pixels[i, j][0], ordinary_pixels[i, j][1], self._find_nearest_value(list_of_values_for_text, ordinary_pixels[i, j][2], format_type), ordinary_pixels[i, j][3]
                    elif format_colour == 'alpha':
                        ordinary_pixels[i, j] = ordinary_pixels[i, j][0], ordinary_pixels[i, j][1], ordinary_pixels[i, j][2], self._find_nearest_value(list_of_values_for_text, ordinary_pixels[i, j][3], format_type)
                else:  # Otherwise
                    if format_colour == 'red':
                        ordinary_pixels[i, j] = self._find_nearest_value(list_of_values_for_bg, ordinary_pixels[i, j][0], format_type), ordinary_pixels[i, j][1], ordinary_pixels[i, j][2], ordinary_pixels[i, j][3]
                    elif format_colour == 'green':
                        ordinary_pixels[i, j] = ordinary_pixels[i, j][0], self._find_nearest_value(list_of_values_for_bg, ordinary_pixels[i, j][1], format_type), ordinary_pixels[i, j][2], ordinary_pixels[i, j][3]
                    elif format_colour == 'blue':
                        ordinary_pixels[i, j] = ordinary_pixels[i, j][0], ordinary_pixels[i, j][1], self._find_nearest_value(list_of_values_for_bg, ordinary_pixels[i, j][2], format_type), ordinary_pixels[i, j][3]
                    elif format_colour == 'alpha':
                        ordinary_pixels[i, j] = ordinary_pixels[i, j][0], ordinary_pixels[i, j][1], ordinary_pixels[i, j][2], self._find_nearest_value(list_of_values_for_bg, ordinary_pixels[i, j][3], format_type)

        self.label_wait.setText("")

        image_file = QFileDialog.getSaveFileName(self, "Save image as...", f"C:\\decrypted_{format_colour}_{format_type}.png",
                                                           'Image (*.png);;Image (*.jpg);;'
                                                           'Image (*.jpeg);;Image (*.bmp);;')[0]
        if not image_file:
            return

        ordinary_image.save(image_file)

    def open_encrypt_image(self) -> None:
        self.file_encrypt_chosen = QFileDialog.getOpenFileName(self, 'Open image...', 'C:\\',
                                                               ';;'.join(f"Image file (*.{i})" for i in EXTENSIONS))[0]
        self.image_lbl_2.setPixmap(QPixmap(self.file_encrypt_chosen).scaled(self.image_lbl_2.width(), self.image_lbl_2.height(), Qt.KeepAspectRatio))
        self.file_encrypt_image = None

        if not self.file_encrypt_chosen:
            return

        *_, self.w, self.h = self._get_image_data(self.file_encrypt_chosen)

    def open_decrypt_image(self) -> None:
        self.file_decrypt_chosen = QFileDialog.getOpenFileName(self, 'Open image...', 'C:\\',
                                                               ';;'.join(f"Image file (*.{i})" for i in EXTENSIONS))[0]
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_chosen).scaled(self.image_lbl.width(), self.image_lbl.height(),  Qt.KeepAspectRatio))
        self.file_decrypt_image = None

    def format_image_by_component(self, format_type: int, component: str) -> None:
        """r = 0, g = 1, b = 2, t = 3; format_type in range(1, 9)"""

        if self.file_decrypt_chosen is None:
            self.make_dialog(f"File is not selected")
            return

        component_index = COMPONENT_INDEX[component]
        image, pixels, x, y = self._get_image_data(self.file_decrypt_chosen)

        format_list = self._generate_format_values(format_type)
        for i in range(x):
            for j in range(y):
                if pixels[i, j][component_index] in format_list:
                    pixels[i, j] = (0, ) * 3 + (pixels[i, j][3], )
                else:
                    pixels[i, j] = (255, ) * 3 + (pixels[i, j][3], )

        self.file_decrypt_image = IMAGE_DECRYPT_FILENAME
        image.save(self.file_decrypt_image)
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def fill_one_colour(self, component: str) -> None:
        """r = 0, g = 1, b = 2"""

        if self.file_decrypt_chosen is None:
            self.make_dialog(f"File is not selected")
            return

        component_index = COMPONENT_INDEX[component]
        image, pixels, x, y = self._get_image_data(self.file_decrypt_chosen)

        for i in range(x):
            for j in range(y):
                data = [0] * 3
                data[component_index] = pixels[i, j][component_index]
                pixels[i, j] = tuple(data) + (pixels[i, j][3], )

        self.file_decrypt_image = IMAGE_DECRYPT_FILENAME
        image.save(self.file_decrypt_image)
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def invert(self) -> None:
        if self.file_decrypt_chosen is None:
            self.make_dialog(f"File is not selected")
            return

        image, pixels, x, y = self._get_image_data(self.file_decrypt_chosen)

        for i in range(x):
            for j in range(y):
                r, g, b, t = pixels[i, j]
                pixels[i, j] = tuple(255 - i for i in (r, g, b)) + (pixels[i, j][3], )

        self.file_decrypt_image = IMAGE_DECRYPT_FILENAME
        image.save(self.file_decrypt_image)
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def make_grey(self) -> None:
        if self.file_decrypt_chosen is None:
            self.make_dialog(f"File is not selected")
            return

        image, pixels, x, y = self._get_image_data(self.file_decrypt_chosen)

        for i in range(x):
            for j in range(y):
                r, g, b, t = pixels[i, j]
                pixels[i, j] = (int(sum((r, g, b)) / 3), ) * 3 + (pixels[i, j][3], )

        self.file_decrypt_image = IMAGE_DECRYPT_FILENAME
        image.save(self.file_decrypt_image)
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def make_bw(self) -> None:
        if self.file_decrypt_chosen is None:
            self.make_dialog(f"File is not selected")
            return

        image, pixels, x, y = self._get_image_data(self.file_decrypt_chosen)

        for i in range(x):
            for j in range(y):
                r, g, b, t = pixels[i, j]
                new_values = (0, ) * 3 if sum((b, g, r)) / 3 < 255 / 2 else (255, ) * 3
                pixels[i, j] = new_values + (pixels[i, j][3], )

        self.file_decrypt_image = IMAGE_DECRYPT_FILENAME
        image.save(self.file_decrypt_image)
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def reflect(self, format_: str) -> None:
        if self.file_decrypt_chosen is None:
            self.make_dialog(f"File is not selected")
            return

        image, *_ = self._get_image_data(self.file_decrypt_chosen)

        self.file_decrypt_image = IMAGE_DECRYPT_FILENAME
        image.transpose({'lr': Image.FLIP_LEFT_RIGHT, 'tb': Image.FLIP_TOP_BOTTOM}[format_]).save(self.file_decrypt_image)
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def reset(self) -> None:
        self.file_decrypt_image = self.file_decrypt_chosen
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def make_randomized(self) -> None:
        if self.file_decrypt_chosen is None:
            self.make_dialog(f"File is not selected")
            return

        image, *_ = self._get_image_data(self.file_decrypt_chosen, None)

        if image.mode not in ("L", "LA", "P", "PA"):
            self.make_dialog(f"Unavailable option for current image mode: {image.mode}")
            return

        image.putpalette(ImagePalette.random("rgb"))

        self.file_decrypt_image = IMAGE_DECRYPT_FILENAME
        image.save(self.file_decrypt_image)
        self.image_lbl.setPixmap(QPixmap(self.file_decrypt_image).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

    def make_dialog(self, message: str) -> None:
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Attention!")
        dlg.setText(message)
        dlg.setWindowIcon(QIcon.fromTheme('warning'))
        dlg.exec()
    
    def resizeEvent(self, event) -> None:
        image_to_show_dec = "" if self.file_decrypt_chosen is None else self.file_decrypt_image if self.file_decrypt_image is not None else self.file_decrypt_chosen
        self.image_lbl.setPixmap(QPixmap(image_to_show_dec).scaled(self.image_lbl.width(), self.image_lbl.height(), Qt.KeepAspectRatio))

        image_to_show_enc = "" if self.file_encrypt_chosen is None else self.file_encrypt_image if self.file_encrypt_image is not None else self.file_encrypt_chosen
        self.image_lbl_2.setPixmap(QPixmap(image_to_show_enc).scaled(self.image_lbl_2.width(), self.image_lbl_2.height(), Qt.KeepAspectRatio))


if __name__ == '__main__':
    sys.excepthook = except_exceptions
    app = QApplication(sys.argv)
    window = StegSolver()
    window.show()
    sys.exit(app.exec())
