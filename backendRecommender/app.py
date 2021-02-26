from flask import Flask, request
from flask_cors import CORS
import json
from Recommender import getRecommendedItems, getAllItems, login
import databaseCon
# db = databaseCon
con = databaseCon.Database();
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/predictions/<string:uid>", strict_slashes=False)
def predictions(uid):
    return json.dumps(getRecommendedItems(uid), indent=2)

@app.route("/products/", strict_slashes=False)
def products():
    return json.dumps(getAllItems(), indent=2)

@app.route("/login", methods = ['POST'])
def login():
    content = request.get_json()
    print(request)
    # return json.dumps(login(user))
    return json.dumps(con.findUser(content), indent=2)
    return 'test';


@app.route("/signup", methods = ['POST'])
def signup():
    content = request.get_json()
    print(request)
    # return json.dumps(login(user))
    return json.dumps(con.createUser(content), indent=2)
    # return 'test';


if __name__ == '__main__':
    app.run(debug=True)