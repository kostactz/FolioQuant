from flask import Flask, request
app = Flask(__name__)
@app.route("/")
def home():
    return request.host.split(':')[0]
if __name__ == '__main__':
    app.run(port=8506)
