import flask
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)

# main index page route
@app.route('/')
def home():
    return '<h1>API is working.. </h1>'


@app.route('/predict',methods=['GET'])
def predict():
    import joblib
    model = joblib.load('carcrash.ml')
    predicted = model.predict([[int(int(request.args['SPEED LIMIT']),
                            int(request.args['AGE']),
                            ,request.args['DRIVER VISION']),
                            int(request.args['DRIVER ACTION']),
                            int(request.args['PHYSICAL CONDITION']),
                            int(request.args['ROADWAY CONDITION']),
                            int(request.args['DEVICE CONDITION']),
                            int(request.args['FIRST CRASH TYPE']),
                           ]])
    return predicted[0]


if __name__ == "__main__":
    app.run(debug=True)
