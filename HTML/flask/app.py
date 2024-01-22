from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def home():
    price = 999
    return render_template('test.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)