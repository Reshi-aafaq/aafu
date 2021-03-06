from flask import Flask ,render_template,url_for,session,redirect
#from first_tf import predct ,model
from wtforms import TextField , SubmitField ,DecimalField
from wtforms.validators import DataRequired,Length,text_type
from wtforms import StringField
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import secrets
import joblib
from flask_wtf import FlaskForm

modell=joblib.load('modeljob.sav')

def predct(data,mymodel):
    #print(data['versicolor'])
    dict={"s_len":data[0],"s_wid":data[1],"p_len":data[2],"p_wid":data[3]}
    dict2={1:"setosa",2:"versicolor",3:"verginica"}
    pdata=mymodel.predict(np.array(data).reshape(1,-1))
    predictions=[]
    return dict,dict2[pdata[0]]



app=Flask(__name__,template_folder='templates')
app.config['SECRET_KEY'] = "aafuu"




class flowerform(FlaskForm):
    sep_len = TextField("Sepal Length",validators=[DataRequired(message='enter value between 1 to 4 digits'),Length(min=1,max=4)])
    sep_width = TextField("Sepal width",validators=[DataRequired(),Length(min=1,max=4)])
    pet_len = TextField("Petal Length",validators=[DataRequired(),Length(min=1,max=4)])
    pet_width = TextField("Petal width",validators=[DataRequired(),Length(min=1,max=4)])

    submit= SubmitField ("Predict")

@app.route('/',methods=['GET', 'POST'])
def index():
    form = flowerform()
    if form.validate_on_submit():
        session['sel_l'] = form.sep_len.data
        session['sep_w'] = form.sep_width.data
        session['pet_l'] = form.pet_len.data
        session['pet_w'] = form.pet_width.data
        return redirect(url_for('predictions'))
    return render_template('irirsflower.html',form=form)




@app.route("/predictions")
def predictions():


    x1= float( session['sel_l'])
    x2= float(session['sep_w'])
    y1= float(session['pet_l'])
    y2= float(session['pet_w'])
    datagen=[x1,x2,y1,y2]
    valuess,flower=predct(datagen,modell)
    return  render_template('predictions.html',reslt=flower.upper())


if __name__ == "__main__":
    app.run(debug=True)


