# app.py
# main function for flask app


# modules
import os
import io
import base64
from flask import Flask, render_template
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__)


# static pages
# home page
@app.route('/')
def home():
    return render_template('home.html')

# function page
@app.route('/functions')
def functions():
    return render_template('functions.html')

# documentations page
@app.route('/documentations')
def documentations():
    return render_template('documentations.html')

# contributions page
@app.route('/contributions')
def contributions():
    return render_template('contributions.html')


# functionalities pages: output text
@app.route('/functions/text_summarization')
def text_summarization():
    return render_template('functionality_text.html', web_title='Text Summarization', content_title='Text Summarization')

@app.route('/functions/text_classification')
def text_classification():
    return render_template('functionality_text.html', web_title='Text classification', content_title='Text Classification')

@app.route('/functions/text_translation')
def text_translation():
    return render_template('functionality_text.html', web_title='Text Translation', content_title='Text Translation')

@app.route('/functions/spelling_correction')
def spelling_correction():
    return render_template('functionality_text.html', web_title='Spelling Correction', content_title='Text Spelling Correction')

@app.route('/functions/high_frequency')
def high_frequency():
    return render_template('functionality_text.html', web_title='High Frequency', content_title='High Frequency Words Substitution')

@app.route('/functions/grammar_check')
def grammar_check():
    return render_template('functionality_text.html', web_title='Grammar Check', content_title='Grammar Check')

@app.route('/functions/punctuation_check')
def punctuation_check():
    return render_template('functionality_text.html', web_title='Punctuation Check', content_title='Punctuation Check')


# functionalities pages: with pictures
# sentiment analysis
# text score
# text statistics

if __name__ == '__main__':
    app.run(debug=True)
