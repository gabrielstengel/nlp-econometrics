import sys
sys.path.append("../")

from flask import Flask, render_template, request

from athena_all import athena

app = Flask(__name__)




client = athena.Athena()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():

    ## Get the user query.
    userQuery = request.args.get('msg')
    resp, result = client.process_query(userQuery)

    return resp


if __name__ == "__main__":
    app.run()
