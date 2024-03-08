from flask import Flask, render_template, request, redirect, url_for
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import wikipedia

app = Flask(__name__)

# GPT-2のトークナイザーとモデルをロード
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    # ユーザーからの入力を取得
    prompt_text = request.form['prompt']
    
    try:
        # Wikipediaからテキストを取得
        wikipedia_text = wikipedia.summary(prompt_text)
        
        # テキストの生成
        inputs = tokenizer.encode(wikipedia_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7)
        
        # 生成されたテキストをデコードしてHTMLコードに組み込む
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 生成されたテキストとWikipediaのテキストと共にHTMLを返す
        return render_template('index.html', prompt_text=prompt_text, generated_text=generated_text, wikipedia_text=wikipedia_text)
    
    except wikipedia.exceptions.DisambiguationError as e:
        # 曖昧性がある場合は、候補のリストを表示
        options = e.options
        return render_template('disambiguation.html', options=options)
    
    except wikipedia.exceptions.PageError:
        wikipedia_text = "Wikipediaにそのトピックが見つかりませんでした。"
        return render_template('index.html', prompt_text=prompt_text, wikipedia_text=wikipedia_text)

@app.route('/generate_with_option/<option>', methods=['GET'])
def generate_with_option(option):
    try:
        # Wikipediaからテキストを取得
        wikipedia_text = wikipedia.summary(option)
        
        # テキストの生成
        inputs = tokenizer.encode(wikipedia_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7)
        
        # 生成されたテキストをデコードしてHTMLコードに組み込む
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 生成されたテキストとWikipediaのテキストと共にHTMLを返す
        return render_template('index.html', prompt_text=option, generated_text=generated_text, wikipedia_text=wikipedia_text)
    
    except wikipedia.exceptions.PageError:
        wikipedia_text = "Wikipediaにそのトピックが見つかりませんでした。"
        return render_template('index.html', prompt_text=option, wikipedia_text=wikipedia_text)

if __name__ == '__main__':
    app.run(debug=True)
