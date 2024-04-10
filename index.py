from flask import Flask, render_template, request, redirect, url_for
from nlp import summarize_cosine, summarize_freq, summarize_luhn

app = Flask(__name__)
text = ''


@app.route('/', methods=['GET', 'POST'])
def home():
    global text
    return render_template('index.html', show=True if text else False, text=text)


@app.route('/summarize', methods=['POST'])
def summarize():
    global text

    if request.form['button'] == 'freq':
        text = ''.join(summarize_freq(request.form['text'])[1])
    elif request.form['button'] == 'luhn':
        text = ''.join(summarize_luhn(request.form['text'])[1])
    else:
        text = ''.join(summarize_cosine(request.form['text'])[1])

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
    