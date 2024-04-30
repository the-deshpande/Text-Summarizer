from flask import Flask, render_template, request, redirect, url_for
from nlp import summarize_cosine, summarize_freq, summarize_luhn

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    
    output = summarize()

    return output


def summarize():

    if request.form['button'] == 'freq':
        text = ''.join(summarize_freq(request.form['text'])[1])
    elif request.form['button'] == 'luhn':
        text = ''.join(summarize_luhn(request.form['text'])[1])
    else:
        text = ''.join(summarize_cosine(request.form['text'])[1])

    return text


if __name__ == '__main__':
    app.run(debug=True)
    