from flask import Flask, render_template, jsonify, send_from_directory, request

from Cancer import XGB_Trained


# COMMENTS TO TEST

#init app and class
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



@app.route("/")
def model_page():
    # Return template and data
    return render_template("ml.html")

@app.route("/makePredictions", methods=["POST"])
def makePredictions():

    genre = int(request.form["genre"])
    age = int(request.form["age"])
    smoke = int(request.form["smoke"])
    alchool = int(request.form["alchool"])
    peer = int(request.form["peer"])
    chronic = int(request.form["chronic"])

    print(genre)
    print(age)
    print(smoke)
    print(alchool)
    print(peer)
    print(chronic)


    prediction = XGB_Trained.XGB_Model(genre, age, smoke, peer, chronic, alchool)
    print("prediction",int(prediction))
    # if int(prediction)==1:
    #     rst="No"
    #
    # elif int(prediction)==2:
    #     rst="Yes"
    score=(genre +smoke*0.7 + alchool*0.4 + age*0.02 + chronic*0.2 + peer*0.3 ) *10


    return render_template("Resultat.html",Result=int(prediction),score=score,genre=genre, age=age, smoke=smoke, peer=peer, chronic=chronic, alchool=alchool)

####################################
# ADD MORE ENDPOINTS

##################YES=2 , NO=1.#########################

#############################################################

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

#main
if __name__ == "__main__":
    app.run(debug=True)
