from flask import Flask,render_template,request,url_for
import random as rdm
app = Flask(__name__,template_folder='templates')
app.static_folder = 'static'
opsTest = ["Vishnu Parammal","Laukik hase","Ajayjumar Pal","Vednarayan Iyer","Vednarayan Iyer" ,"Sagar Udasi","Chetan Borse","Vaibhav Chavhan","Akhilesh Mehar"]
Labels = ["Maintainance","Banking","Vehicle","Technical Help","Hardware Damage"]
Context = ["Finance","Banking and Economy","Efficiency","Complaint","Satisfactory","Result"]
Emotion = ["Angry","Sad","Neural","Calm","Happy","Surpirsed","Disgust"]
opHolder='<tr><th scope="row">{}</th><td><a href="{}"><button type="button" class="btn btn-light">{}</button></a></td></tr>'

@app.route('/')
def home():
    global opHolder,opsTest
    finalHTML=""
    for i in range(1,len(opsTest)+1):
        finalHTML+=opHolder.format(1,url_for('operator',op=opsTest[i-1]),opsTest[i-1])

    return render_template("index.html",ops=finalHTML)


@app.route('/operator',methods=['GET'])
def operator():
    #  Label,call_duration,context,emotion
    global opHolder
    op = request.args.get('op')

    print("Operator:",op)
    return render_template("index2.html",opName=op,Label=rdm.choice(Labels),call_duration=rdm.randrange(1,10,1),context=rdm.choice(Context),emotion=rdm.choice(Emotion))

if __name__ == '__main__':
    app.run()