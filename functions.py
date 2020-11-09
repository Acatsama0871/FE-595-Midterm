# functions.py
# the function for NLP


# modules
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from collections import Counter
from flask import Markup
from textblob import TextBlob, Word
from textblob.wordnet import VERB, ADV, ADJ
from nltk import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from tabulate import tabulate
from gingerit.gingerit import GingerIt
from gensim import corpora, models
from gensim.summarization import summarize
from gensim.summarization import keywords
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


# sample function
def Sample(Str):
    doc1 = "New Zealand is a small country of four million inhabitants, a long-haul flight from all the major tourist-generating markets of the world." \
           " Tourism currently makes up 9% of the country’s gross domestic product, and is the country’s largest export sector. " \
           "Unlike other export sectors, which make products and then sell them overseas, tourism brings its customers to New Zealand. " \
           "The product is the country itself - the people, the places and the experiences. " \
           "In 1999, Tourism New Zealand launched a campaign to communicate a new brand position to the world. " \
           "The campaign focused on New Zealand’s scenic beauty, exhilarating outdoor activities and authentic Maori culture, and it made New Zealand one of the strongest national brands in the world."

    doc2 = "I lov my dog"

    doc3 = 'Show is bad. Phone is bad. Photo is bad. ' \
           'Text is good. Cake is good. Smile is good. '

    doc4 = '''Do not go gentle into that better night,
    Old age shoulds burn and rave at close of day;
    Rage, rage against the dying of the light.'''

    doc5 = "cat is on the mat"

    doc6 = "dog is in the fog"

    doc7 = "The smelt of fliwers bring bcak memories."

    if Str == 'Text_Summarization':
        return doc1
    elif Str == 'Text_Classification':
        return doc1
    elif Str == 'Text_Translation':
        return doc1
    elif Str == 'Text_Spelling_Correction':
        return doc2
    elif Str == 'Sentiment_Analysis':
        return doc1
    elif Str == 'Synonymous_Substitution':
        return doc3
    elif Str == 'Grammar_Check':
        return doc4
    elif Str == 'Sentence_Similarity':
        return doc5, doc6
    elif Str == 'Text_Score':
        return doc7
    elif Str == 'Text_Statistics':
        return doc1


# text summarization
def text_summarization(text):
    # tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)

    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(i) for i in tokens]

    # delete the stop words
    stop_words = set(stopwords.words('english'))
    stopped_tokens = [i for i in lemma if not i in stop_words]
    texts = [stopped_tokens]

    # constructing the document-term
    dictionary = corpora.Dictionary(texts)  # !!!!!!!!!!!!!

    # bag of the words
    corpus = [dictionary.doc2bow(text) for text in texts]  # !!!!!
    themodel = models.ldamodel.LdaModel(corpus, num_topics=1,
                                        id2word=dictionary, passes=5)

    # Keywords group1
    topic1 = np.array(themodel.top_topics(corpus)[0][0])[:, 1]
    topic1 = 'The recommended keys words(1): ' + ",".join(topic1)

    # Keywords group2
    topic2 = np.array(keywords(text, words=24).split("\n"))
    topic2 = 'The recommended keys words: ' + ",".join(topic2)

    # Summarization
    summarization = summarize(text, word_count=40)
    summarization = "The main idea for this topic:<br>" + summarization

    # string = Markup(summarization + '<br><br>' + topic1 + '<br><br>' + topic2)
    string = Markup(summarization + '<br><br>' + topic2)
    return string


# text classification
def text_classification(text):
    current = os.getcwd()
    new = current + "/Model" + "/model.h5"  # TODO change path
    model = keras.models.load_model(new)

    os.chdir(current + "/Tokenizer")  # TODO change dir
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)
    os.chdir(current)

    seq = x_tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict_classes(padded)
    labels = ['Sport', 'Business', 'Politics', 'Tech', 'Entertainment', 'Others']
    
    return_string = 'Text Class:' + '<br><br>' + labels[pred[0]]

    return Markup(return_string)


# text translation
def text_translate(text, language):
    try:
        # Input
        input_ = TextBlob(text)

        # Output
        output = input_.translate(to=language)
        res = str(output)

    except:
        res = "Please enter the language you want to translate again"

    return res


# text spelling correction
def text_spelling_correction(text):
    # Spelling correction
    text = TextBlob(text)
    before = text.words
    after = text.correct().words

    # Get the location of every changed word
    resultnum = []
    resultbef = []
    resultaft = []
    for i in range(0, len(before) - 1):
        if before[i] != after[i]:
            resultnum.append(str(i + 1))
            resultbef.append(str(before[i]))
            resultaft.append(str(after[i]))

    res1 = str(text.correct())
    res2 = "/".join(resultbef)
    res3 = "/".join(resultaft)
    res = Markup(res1)
    
    return res


# sentiment analysis
def text_sentiment_analysis(text):
    blob = TextBlob(text)
    score1 = np.array([])
    score2 = np.array([])
    for sentence in blob.sentences:
        score1 = np.append(score1, sentence.sentiment.polarity)
        score2 = np.append(score2, sentence.sentiment.subjectivity)  # TODO validate
    ave1 = np.mean(score1)
    ave2 = np.mean(score2)
    string = 'The average polarity score for this text is: ' + str(ave1) + '<br>' + \
             'The average subjective score for this text is ' + str(ave2) + '<br><br>' + \
             'Sentence Index' + '&emsp;&emsp;' + 'Polarity Score' + '&emsp;&emsp;' + 'Subjectivity' + '<br>'

    range = np.arange(0, len(score1), 1)
    for i in range:
        string = string + "Sentence" + str(i) + '&emsp;&emsp;&emsp;&emsp;' + '%.5f' % round(score1[i], 4) + '&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;'\
                 + '%.5f' % round(score2[i], 4) + '<br>'

    return Markup(string)


# synonymous substitution
def tag_converter(tag):
    if tag in ("JJ", "JJR", "JJS"):
        return ADJ
    if tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return ADV
    if tag in ("RB", "RBR", "RBS"):
        return VERB


def text_synonymous_substitution(text, freq_limit=3):
    text = text.replace('the', '')
    text = text.replace('The', '')
    sentences_blob = TextBlob(text)
    sentences_count = 1
    dictionary = []

    try:
        for cur_sentence in sentences_blob.sentences:
            blob_words = cur_sentence.tags
            cur_dictionary = [
                {'word_original': str(blob_words[i][0]), 'tag_original': blob_words[i][1], 'word_pos': i + 1,
                 'sentence_pos': sentences_count} for i in range(len(blob_words))]
            for cur_record in cur_dictionary:
                cur_lemma = Word(cur_record['word_original']).lemmatize(cur_record['tag_original'])
                cur_record['word_lemma'] = str(cur_lemma)

            dictionary.extend(cur_dictionary)
            sentences_count += 1

        # delete: [am, is, are] and only keep verb, adj and adv
        delete_index = []
        for i in range(len(dictionary)):
            if dictionary[i]['word_lemma'] == 'be' or (dictionary[i]['tag_original'] not in ['JJ', 'JJR',
                                                                                             'JJS']):  # 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
                delete_index.append(i)
        dictionary = [i for j, i in enumerate(dictionary) if
                      j not in delete_index]  # Stack overflow:https://stackoverflow.com/questions/497426/deleting-multiple-elements-from-a-list

        # found the word beyond limit
        # Compute the frequency of each element.
        words_lemma = [i['word_lemma'] for i in dictionary]
        count = dict(Counter(words_lemma))
        selected = []
        for cur_word in count:
            if count[cur_word] >= 3:
                selected.append(cur_word)

        string_formatter = "sentence {} word {}, "
        dictionary_selected = []
        for cur_word in selected:
            cur_record = {'Word_lemma': cur_word}
            cur_string = ""
            for record in dictionary:
                if record['word_lemma'] == cur_word:
                    cur_string += string_formatter.format(record['sentence_pos'], record['word_pos'])
                    cur_original = record['word_original']
                    cur_tag = record['tag_original']
            cur_record['string'] = cur_string
            cur_record['Word_original'] = cur_original
            cur_record['tag'] = cur_tag
            dictionary_selected.append(cur_record)

        # found synonym
        for cur_record in dictionary_selected:
            cur_word = cur_record['Word_lemma']
            cur_tag = tag_converter(cur_record['tag'])
            cur_result = []

            for synset in Word(cur_word).get_synsets(pos=cur_tag):
                for lemma in synset.lemmas():
                    cur_result.append(lemma.name())
            cur_result = list(set(cur_result))
            if len(cur_result) > 5:
                cur_result = cur_result[0:5]
            synonym_string = ",".join(cur_result)
            cur_record['synonym'] = synonym_string

        # output string
        output_string_formatter = 'At {}, the word "{}" can be replaced by [{}]<br>'
        if len(dictionary_selected) == 0:
            output_string = "No suggestion"
        else:
            output_string = "Suggestions:<br><br>"
            for cur_record in dictionary_selected:
                output_string += output_string_formatter.format(cur_record['string'],
                                                                cur_record['Word_original'],
                                                                cur_record['synonym'])

        return Markup(output_string)
    except:
        return "Sorry some words may not in the corpus."


# grammar check
def text_grammar_check(text):
    # initialize
    parser = GingerIt()
    corrected_text_string = ''
    corrected_words_string = ''
    changes_count = 0
    formatter_words = 'At sentence {} word {}, the "{}" is replaced by "{}".<br>'
    header_string = '{} change(s) are made to original text. <br><br> The changed text is:<br><br>'

    try:
        # sentences tokenization
        sentence_blob = TextBlob(text)
        sentences_count = 1

        for cur_sentence in sentence_blob.sentences:
            # parse text
            parsed = parser.parse(str(cur_sentence))
            parsed_text = parsed['result']
            parsed_corrections = parsed['corrections']
            parsed_corrections = parsed_corrections[::-1]

            # words in original sentence
            blob = TextBlob(text)
            original_words = [i for i in blob.words]

            # changed words
            corrected_words = []
            original_text = []
            for item in parsed_corrections:
                original_text.append(item['text'])
                corrected_words.append(item['correct'])

            changed_index = []
            for i in original_text:
                changed_index.append(original_words.index(i))

            # string result
            corrected_text_string += parsed_text
            for i in range(len(corrected_words)):
                corrected_words_string += formatter_words.format(sentences_count,
                                                                 i + 1,
                                                                 original_text[i],
                                                                 corrected_words[i])
                changes_count += 1

            sentences_count += 1

        # output string
        corrected_text_string += '<br><br>'
        output_string = header_string.format(changes_count) + corrected_text_string + corrected_words_string
    except:
        return "Sorry, the function can't handle the text. Please try another one."

    return Markup(output_string)


# sentence similarity
def text_sentence_similarity(sentence1, sentence2):
    # embedding
    embed = hub.load("https://tfhub.dev/google/nnlm-en-dim50/2")
    vector1 = embed([sentence1]).numpy()
    vector2 = embed([sentence2]).numpy()

    # calculate cos
    cos = np.sum(vector1 * vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    return_string = 'cos similarity:<br><br>' + str(cos)

    return Markup(return_string)


# text score
def text_score(text):
    # initialize
    parser = GingerIt()
    grammar_count = 0
    spelling_count = 0
    grammar_string = 'Grammar error: '
    spelling_string = 'Spelling error: '

    # check error
    try:
        # sentences tokenization
        sentence_blob = TextBlob(text)

        for cur_sentence in sentence_blob.sentences:
            # spelling error
            for cur_word in cur_sentence.words:
                corrected = cur_word.correct()
                if corrected != cur_word:
                    spelling_string += str(cur_word) + ', '
                    spelling_count += 1
            cur_sentence = cur_sentence.correct()

            # grammar error
            parsed = parser.parse(str(cur_sentence))
            grammar_count += len(parsed['corrections'])
            for item in parsed['corrections']:
                grammar_string += item['text'] + ', '

    except:
        return "Please try another text."

    # calculate score
    the_score = 100 - 3 * grammar_count - 3 * spelling_count
    if the_score < 0:
        the_score = 0

    # return string
    return_string_formatter = "The score of the text is {}.<br>(Each grammar error: -3, Each spelling error: -3))<br><br>"
    return_string_formatter += grammar_string + '<br>' + spelling_string
    return_string = return_string_formatter.format(the_score)

    return Markup(return_string)


# text statistics
def text_statistics_wordTag(doc):

    tokenizer = RegexpTokenizer(r'\w+')
    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)

    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(i) for i in tokens]

    # delete the stop words
    stop_words = set(stopwords.words('english'))
    stopped_tokens = [i for i in lemma if not i in stop_words]
    text = stopped_tokens

    # Tag frequency
    count = pos_tag(text)
    word_list = pd.DataFrame(count, columns=['Word', "Tag"])
    Tag_fre = word_list['Tag'].value_counts()
    Tag_fre = Tag_fre.reset_index()

    # The average number of words per sentence
    number_sentence = len(sent_tokenize(doc))
    number_words = len(word_tokenize(doc))
    Theaverage = float(number_words / number_sentence)
    string = "The average number of words per sentence: " + str(Theaverage) + "<br><br><br>" + \
             "Tag" + "&emsp;&emsp;&emsp;" + "Count" + "&emsp;&emsp;&emsp;" + "Frequency" + "<br>"

    for i in range(len(Tag_fre)) :
        string = string + '<br>' + '%4s' % Tag_fre['index'][i] + "&emsp;&emsp;&emsp;" + '%3d' % Tag_fre['Tag'][i] + "&emsp;&emsp;&emsp;" + '%.4f' % (Tag_fre['Tag'][i] / number_words)

    string = Markup(string)

    return string
