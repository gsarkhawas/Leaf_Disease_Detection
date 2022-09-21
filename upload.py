import os
from flask import Flask, request, send_from_directory, render_template
from ModelTest import classifyImage

UPLOAD_FOLDER = '/home/sunbeam/Guzzu-DBDA/Project/Images/Potato_Try'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','JPG'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            imagefullpath=os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(imagefullpath)
            print(f"INFO: filename sent for classifcation : {imagefullpath}")
            prediction = classifyImage(imagefullpath)
            return render_template('template.html', filename=file.filename, prediction=prediction)
    return '''
    <!doctype html>
    <title>Image-Based Plant Disease Detection Using Deep Learning</title>
    <h1>
        <div class="panel panel-primary"><span class="label label-info">Kindly upload plant image</span></div>
    </h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>.filename
         <input type=submit value=Upload>
    </form>
    
    '''

@app.route('/uploads/<filename>')
def send_file(filename):
    print(f"INFO : {filename} being sent from directory {UPLOAD_FOLDER}")
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host= 'localhost', port=8080)