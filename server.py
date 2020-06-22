from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

# model imports
from recipe_reverseengin import RE_recipes

app = Flask(__name__)
api = Api(app)

CORS(app)

@app.route("/")
def initialization():

    # Image reverse-engineering test
    information = RE_recipes(img_path = './recipe_generation/data/demo_imgs/1.jpg')

    # Image style transfer test

    # return jsonify({'text':'🤖 local server running 🤖'})
    return jsonify({'text':str(information)})


class Employees(Resource):
    def get(self):
        return {'employees': [{'id':1, 'name':'Balram'},{'id':2, 'name':'Tom'}]} 

class Employees_Name(Resource):
    def get(self, employee_id):
        print('Employee id:' + employee_id)
        result = {'data': {'id':1, 'name':'Balram'}}
        return jsonify(result)       


api.add_resource(Employees, '/employees') # Route_1
api.add_resource(Employees_Name, '/employees/<employee_id>') # Route_3


if __name__ == '__main__':
     app.run(port=5002)
