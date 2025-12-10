import os

from flask import Flask, send_from_directory

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(BASE, "source_code")
RESULTS = os.path.join(BASE, "results")

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(SRC, "main.html")


@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS, filename)


if __name__ == "__main__":
    app.run(debug=True)
