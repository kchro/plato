#!/usr/bin/env python

from flask import Flask, request, session, render_template
from flask_cors import CORS, cross_origin
import subprocess

import json
import random

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def hello():
  return "<h1>Hi!</h1>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
#    app.run()
