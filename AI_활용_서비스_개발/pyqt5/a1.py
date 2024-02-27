import sys
from PyQt5.QtWidgets import QApplication, QWidget,QLabel, QDesktopWidget
from PyQt5.QtGui import QPixmap

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.center()
        
    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 0)

        label1 = QLabel('aaa', self)
        label1.move(0,10)

        label2 = QLabel(self)
        pixmap = QPixmap('pic1.png')

        label2.move(0,30)
        label2.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        self.setGeometry(500, 200, 500, 400)
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
   app = QApplication(sys.argv)
   print(len(sys.argv))
   print(sys.argv[0])
   ex = MyApp()
   sys.exit(app.exec_())