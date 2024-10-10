from flask import Flask, render_template, request
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    # TODO: Write the code that calls the sentiment analysis functions here.
    # hint: use request.method == "POST"
    return render_template('form.html')
if __name__ == "__main__":
    app.run()
