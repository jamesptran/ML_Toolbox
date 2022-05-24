from flask import Flask, render_template, request, Response


app = Flask(__name__)

@app.route("/")
def test():
    return "<p>Hello, World!</p>"


@app.route("/home")
def home():
    return render_template('home.html')
