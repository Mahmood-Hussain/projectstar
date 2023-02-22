from flask import Flask
from dotenv import load_dotenv
import sys

sys.path.append('/methods')

app = Flask(__name__, template_folder='../templates', static_folder='../static')
load_dotenv()

from app import routes
