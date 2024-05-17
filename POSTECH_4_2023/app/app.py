import json
import logging
from flask import Flask, request, jsonify
from agent import Service
import sys

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_service = Service()

@app.route('/')
def run_test():
    return "App works!"

@app.route('/api/task_name', methods=['GET'])
def task_name():
    return api_service.get_task_name()

@app.route('/api/run', methods=['POST'])
def run_task():
    content = request.get_json(silent=True, cache=False, force=True)
    return api_service.run(content)