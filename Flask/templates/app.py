

#pip install flask
#pip install prsaw
#pip install assistant 

from flask import Flask, jsonify, render_template, request



#initialising the class using flask
appli = Flask(__name__)


@appli.route("/")
@appli.route("/home")

#returns this 
def home():
    return render_template("homepage.html")

@appli.route("/FAQ")
def about():
    return render_template("FAQ.html")


@appli.route("/contactOrganisations")
def about():
    return render_template("contactOrganisations.html")
