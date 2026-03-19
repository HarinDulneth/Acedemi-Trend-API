# app.py

from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)
VIS_DIR = os.path.join("visualizations", "templates")

# Route to list all HTML visualizations
@app.route("/")
def index():
    files = [f for f in os.listdir(VIS_DIR) if f.endswith(".html")]
    return render_template("index.html", files=files)

# Route to serve individual HTML visualization
@app.route("/view/<filename>")
def view_file(filename):
    return send_from_directory(VIS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
