from fastapi import FastAPI
from hf_sentiment import sentiment_analysis, collect_data
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.get('/')
async def root():
    return {'example': 'This is an example', 'data': 0}


@app.get('/sentiment_analysis/{sentence}')
async def get_analysis(sentence: str):
    result = sentiment_analysis(sentence)
    return result

@app.get('/add_to_datasets/{sentence}')
async def get_datasets(sentence: str):
    result = collect_data(sentence)
    if result != '':
        return True
    else:
        return False