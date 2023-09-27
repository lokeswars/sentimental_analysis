from flask import Flask, render_template, request, redirect, url_for
import preprocessing
import joblib,pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        submitted_text = request.form['text']
        #text="#inida @kingjan follow me to welcome"
        ct=preprocessing.clean_text(submitted_text)
        stop_free=preprocessing.stop_free(ct)
        vectorized = preprocessing.vec(pd.DataFrame([stop_free], columns=['cleaned_tweet']))
        model = joblib.load('model.pkl')
        rf = model.predict(vectorized)

        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(submitted_text)
        sentiment_score = sentiment_scores['compound']

        
        if rf==0:
            r="Positive Post"
        else:
            r="Negative Post"
        return render_template('index.html', submitted_text=submitted_text, sentiment_label=r, sentiment_score=sentiment_score)
    

if __name__ == '__main__':
    app.run(debug=True)
