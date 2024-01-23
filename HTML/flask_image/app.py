from flask import Flask, render_template

app = Flask(__name__)

# 이미지 파일 경로 리스트
image_paths = [
    "static/images/image1.jpg",
    "static/images/image2.jpg",
    "static/images/image3.jpg",
    "static/images/image4.jpg",
    "static/images/image5.jpg",
    "static/images/image6.jpg"
]

# 이미지를 3x3 그리드로 나누기
num_columns = 3
image_grid = [image_paths[i:i + num_columns] for i in range(0, len(image_paths), num_columns)]

# 라우트 정의
@app.route('/')
def index():
    return render_template('index.html', image_grid=image_grid)

if __name__ == '__main__':
    app.run(debug=True)
