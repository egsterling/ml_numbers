import flask
from evaluate_image import predict, predictDict
import numpy as np

# IMPLEMENT DATA AUGMENTATION WITH TRANSLATION AND CHANGING SIZE
# TRY CHANGING NEURAL NETWORK ARCHITECTURE

app = flask.Flask(__name__)

IMG_PATH = "data/mnist_png/Hnd/Sample4/53.png"
INPUT_EPOCH = 11

@app.route("/api/value/", methods=["POST"])
def get_val():
    img = np.transpose(np.array(flask.request.json)) * (256.0/100.0)
    prediction = predictDict(img, INPUT_EPOCH)
    dic = {i: prediction[i].item() for i in range(10)}
    print(dic)
    return flask.jsonify(dic)

@app.route("/")
def show_index():
    return flask.send_file('index.html')

@app.route('/style.css')
def serve_css():
    return flask.send_file('style.css', mimetype='text/css')

@app.route('/frontend.js')
def serve_js():
    return flask.send_file('frontend.js', mimetype='text/javascript')

if __name__ == "__main__":
    app.run(host='localhost', port=5500, debug=True)
