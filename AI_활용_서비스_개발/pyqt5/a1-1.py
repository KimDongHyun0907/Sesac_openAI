import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from module import menu
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        line=QLabel('-'*100,self)
        line.move(0, 0)
        line=QLabel('-'*100,self)
        line.move(0, 80)
        
        label1 = QLabel(menu.info, self)
        label1.move(5, 10)
       
        self.setWindowTitle('python PyQt5')
        self.setGeometry(500, 200, 800, 900)
        
        btn1 = QPushButton("load",self)
        btn1.setFixedSize(200, 30)  # 버튼의 폭과 높이를 설정
        btn1.move(300,100)  # btn1의 위치를 도형의 정가운데에 넣고자 한다면
        btn1.clicked.connect(self.showImage)  # 버튼 클릭 이벤트 연결

        self.label_image = QLabel(self)  # 이미지를 표시할 라벨 생성
        self.label_image.setGeometry(50, 150, 700, 700)  # 라벨의 위치와 크기 설정
        self.label_image.setAlignment(Qt.AlignHCenter)  # 이미지를 중앙 정렬로 설정

        self.show()

    def showImage(self):
        # 이미지 파일 경로
        image_path = 'pic1.png'
        pixmap = QPixmap(image_path)  # QPixmap을 사용하여 이미지 로드
        self.label_image.setPixmap(pixmap)  # 라벨에 이미지 표시

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
