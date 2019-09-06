from flask import Flask,request
import os

app = Flask(__name__)
from app import views