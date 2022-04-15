

#pip install flask
#pip install prsaw
#pip install assistant 
#pip install waitress

from flask import Flask,  render_template, request


#initialising the class using flask
app = Flask(__name__)

@app.route("/")
#returns this 
def home():
    return render_template("homepage.html")

@app.route("/contactOrgansiations")
#returns this 
def contactOrgansiations():
    return render_template("contactOrgansiations.html")

@app.route("/FAQ")
#returns this 
def FAQ():
    return render_template("FAQ.html")

if __name__ == "__main__":
    app.run(debug=True)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)

