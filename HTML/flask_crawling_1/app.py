from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    user = '반원'
    data = {'level': 60, 'point': 360, 'exp' : 45000}
    return render_template('index.html', user = user, data = data)

if __name__=="__main__":
    app.run(debug=True)