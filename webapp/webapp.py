import csv
from io import StringIO

import numpy as np
from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from joblib import load
from wtforms import RadioField
from wtforms.validators import DataRequired

app = Flask(__name__)

cancer_choices = [
    'Brain Cancer',
    'Breast Cancer',
    'Liver Cancer',
    'Prostate Cancer'
]


class PredictionForm(FlaskForm):
    class Meta:
        csrf = False

    input_file = FileField(validators=[FileRequired(), FileAllowed(['csv'], 'CSV only!')])
    cancer_type = RadioField(choices=cancer_choices, validators=[DataRequired()])


@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    prediction = None

    if form.validate_on_submit():
        file = form.input_file.data
        cancer_type = form.cancer_type.data

        # Read the data from the input file
        data = []
        for lines in csv.reader(StringIO(file.read().decode()), delimiter=','):
            data.append(lines[-1])

        # Load the trained Linear Discriminant Analysis (LDA) model
        classifier = load('../joblibs/' + cancer_type + '.joblib')

        # Load the indexes of features selected by Harmony Search (HS)
        idx = load('../joblibs/' + cancer_type + '-idx.joblib')

        try:
            # Perform feature selection
            # Select only those features that were selected by HS while training the model
            data = np.array([data])
            data = data[:, idx]
        except IndexError:
            # User has provided invalid input data for the selected cancer type
            prediction = 'invalid to predict ' + cancer_type
        else:
            prediction = 'Cancerous' if classifier.predict(data)[0] else 'Non Cancerous'

    return render_template('index.html', form=form, prediction=prediction)


if __name__ == '__main__':
    app.run()
