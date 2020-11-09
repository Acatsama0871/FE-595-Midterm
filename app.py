# app.py
# main function for flask app


# modules
import os
import io
import base64
from flask import Flask, render_template, request
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import functions as f


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
    sample = f.Sample('Text_Summarization')

    return render_template('functionality_text.html', 
                           web_title='Text Summarization', 
                           content_title='Text Summarization',
                           action_content='/functions/text_summarization/submit',
                           sample_content=sample,
                           result_content='')

@app.route('/functions/text_summarization/submit', methods=['POST'])
def text_summarization_submit():
    if request.method =='POST':
        try:
            text_ = request.form['input_text']
            result = f.text_summarization(text=text_)
            
            return render_template('functionality_text.html', 
                                web_title='Text Summarization', 
                                content_title='Text Summarization',
                                action_content='/functions/text_summarization/submit',
                                sample_content=text_,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')


@app.route('/functions/text_classification')
def text_classification():
    sample = f.Sample('Text_Classification')

    return render_template('functionality_text.html', 
                           web_title='Text Classification', 
                           content_title='Text Classification',
                           action_content='/functions/text_classification/submit',
                           sample_content=sample,
                           result_content='')
    
@app.route('/functions/text_classification/submit', methods=['POST'])
def text_classification_submit():
    if request.method =='POST':
        try:
            text_ = request.form['input_text']
            result = f.text_classification(text=text_)
            
            return render_template('functionality_text.html', 
                                web_title='Text Classification', 
                                content_title='Text Classification',
                                action_content='/functions/text_classification/submit',
                                sample_content=text_,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')


@app.route('/functions/text_translation')
def text_translation():
    sample = f.Sample('Text_Translation')

    return render_template('functionality_text.html', 
                           web_title='Text Translation', 
                           content_title='Text Translation',
                           action_content='/functions/text_translation/submit',
                           sample_content=sample,
                           result_content='')
    
@app.route('/functions/text_translation/submit', methods=['POST'])
def text_Translation_submit():
    if request.method == 'POST':
        try:
            text_ = request.form['input_text']
            result = f.text_translate(text=text_, language='fr')
            
            return render_template('functionality_text.html', 
                                web_title='Text Translation', 
                                content_title='Text Translation',
                                action_content='/functions/text_translation/submit',
                                sample_content=text_,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')


@app.route('/functions/spelling_correction')
def spelling_correction():
    sample = f.Sample('Text_Spelling_Correction')

    return render_template('functionality_text.html', 
                           web_title='Spelling Correction', 
                           content_title='Spelling Correction',
                           action_content='/functions/spelling_correction/submit',
                           sample_content=sample,
                           result_content='')
    
@app.route('/functions/spelling_correction/submit', methods=['POST'])
def spelling_correction_submit():
    if request.method == 'POST':
        try:
            text_ = request.form['input_text']
            result = f.text_spelling_correction(text=text_)
            
            return render_template('functionality_text.html', 
                                web_title='Spelling Correction', 
                                content_title='Spelling Correction',
                                action_content='/functions/spelling_correction/submit',
                                sample_content=text_,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')

@app.route('/functions/high_frequency')
def high_frequency():
    sample = f.Sample('Synonymous_Substitution')

    return render_template('functionality_text.html', 
                           web_title='Words Substitution', 
                           content_title='Words Substitution',
                           action_content='/functions/high_frequency/submit',
                           sample_content=sample,
                           result_content='')
    
@app.route('/functions/high_frequency/submit', methods=['POST'])
def high_frequency_submit():
    if request.method == 'POST':
        try:
            text_ = request.form['input_text']
            result = f.text_synonymous_substitution(text=text_)
            
            return render_template('functionality_text.html', 
                                web_title='Words Substitution', 
                                content_title='Words Substitution',
                                action_content='/functions/high_frequency/submit',
                                sample_content=text_,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')

@app.route('/functions/grammar_check')
def grammar_check():
    sample = f.Sample('Grammar_Check')

    return render_template('functionality_text.html', 
                           web_title='Grammar Check', 
                           content_title='Grammar Check',
                           action_content='/functions/grammar_check/submit',
                           sample_content=sample,
                           result_content='')

@app.route('/functions/grammar_check/submit', methods=['POST'])
def grammar_check_submit():
    if request.method == 'POST':
        try:
            text_ = request.form['input_text']
            result = f.text_grammar_check(text=text_)
            
            return render_template('functionality_text.html', 
                                web_title='Grammar Check', 
                                content_title='Grammar Check',
                                action_content='/functions/grammar_check/submit',
                                sample_content=text_,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')

@app.route('/functions/sentences_similarity')
def sentences_similarity():
    sample1_, sample2_ = f.Sample('Sentence_Similarity')
    
    return render_template('functionality_sentence.html', 
                           web_title='Sentences Similarity', 
                           content_title='Sentences Similarity',
                           action_content='/functions/sentences_similarity/submit',
                           sample1=sample1_,
                           sample2=sample2_,
                           result_content='')

@app.route('/functions/sentences_similarity/submit', methods=['POST'])
def sentences_similarity_submit():
    if request.method == 'POST':
        try:
            sen1 = request.form['input_text1']
            sen2 = request.form['input_text2']
            result = f.text_sentence_similarity(sen1, sen2)
            
            return render_template('functionality_sentence.html', 
                                web_title='Sentences Similarity', 
                                content_title='Sentences Similarity',
                                action_content='/functions/sentences_similarity/submit',
                                sample1=sen1,
                                sample2=sen2,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')


@app.route('/functions/text_score')
def text_score():
    sample = f.Sample('Text_Score')

    return render_template('functionality_text.html', 
                           web_title='Text Score', 
                           content_title='Text Score',
                           action_content='/functions/text_score/submit',
                           sample_content=sample,
                           result_content='')

@app.route('/functions/text_score/submit', methods=['POST'])
def text_score_submit():
    if request.method == 'POST':
        try:
            text_ = request.form['input_text']
            result = f.text_score(text=text_)
            
            return render_template('functionality_text.html', 
                                web_title='Text Score', 
                                content_title='Text Score',
                                action_content='/functions/text_score/submit',
                                sample_content=text_,
                                result_content=result)
        except :
            return render_template('home.html')
    else:
        return render_template('home.html')


# functionalities pages: with pictures
# sentiment analysis
# text statistics


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
