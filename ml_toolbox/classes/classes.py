from flask import g, Flask, current_app
import mongoengine
from unite.core import db
import json
import math
from datetime import datetime, timedelta
from unite.utils.datetools import *
import warnings
from unite.awareconfig import default_sensors, default_plugins, host_name



class User(db.Document, UserMixin):
    name = db.StringField()
    email = db.StringField(max_length=255, unique=True)
    id_token = db.StringField(max_length=255)

    extra_info = db.DynamicField(default=[])

    def __eq__(self, other):
        return self.email == other.email

class Model(db.Document):
    model_id = db.StringField(max_length=255, unique=True)
    model_path = db.StringField()
    data_path = db.StringField()
