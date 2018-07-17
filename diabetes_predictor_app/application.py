import flask
from sklearn import linear_model
from sklearn import pipeline, preprocessing, neighbors, model_selection
import numpy as np
import pandas as pd

#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on breast cancer survival,
# Build a LogisticRegression predictor on it
df = pd.read_csv('diabetes_pima.csv')
columns_lower=[x.lower() for x in df.columns]
df.columns=columns_lower

x = df[['glucose','bloodpressure','insulin','bmi','age']]
y=df['outcome']

logr_pipe=pipeline.Pipeline([
    ('scaler',preprocessing.StandardScaler()),
    ('logr',linear_model.LogisticRegression())
])

PREDICTOR = logr_pipe.fit(x,y)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
application = flask.Flask(__name__)

# Homepage
@application.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("index.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@application.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data["example"])
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
# application.run(host='0.0.0.0')
# application.run(debug=True)

if __name__ == "__main__":
  application.run()
