from flask import Flask
from flask_cors import CORS
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

from routes import register_routes
register_routes(app)

if __name__ == '__main__':
    app.run(debug=True, port=5000)