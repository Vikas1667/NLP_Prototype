from io import BytesIO
import base64
from fastapi import FastAPI
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
import nltk
from nltk.tokenize import sent_tokenize
# from gensim.summarization import summarize
from wordcloud import WordCloud, STOPWORDS 
from transformers import T5Tokenizer, T5ForConditionalGeneration

def t5_summarizer(text):
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_text = "summarize: " + text
    # t5_text
    inputs = t5_tokenizer.encode(t5_text, return_tensors="pt", max_length=512, padding='max_length', truncation=True)
    summary_ids = t5_model.generate(inputs, num_beams=int(2),
                                    no_repeat_ngram_size=3,
                                    length_penalty=2.0,
                                    min_length=100,
                                    max_length=200,
                                    early_stopping=True)

    output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output

app = FastAPI()

templates = Jinja2Templates(directory="templates")


nltk.download('punkt') # download this
@app.get("/")
def home(request: Request):
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def home(request: Request):
    sumary=""
    if request.method == "POST": 
        form = await request.form()
        if form["message"] and form["word_count"]: 
            word_count = form["word_count"]
            text = form["message"]
            sumary=t5_summarizer(text)
            word_cloud = wordcloud(sumary)

            return templates.TemplateResponse("index.html", {"request": request, "sumary": sumary, "wordcloud": word_cloud})



def wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(text).to_image()
    img = BytesIO()
    wordcloud.save(img, "PNG")
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()
    return img_b64

    