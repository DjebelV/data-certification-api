
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd


app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness,danceability,duration_ms,energy,explicit,id,instrumentalness,key,liveness,loudness,mode,name,release_date,speechiness,tempo,valence,artist):

    x_pred={'acousticness':[float(acousticness)],
    'danceability':[float(danceability)],
    'duration_ms':[int(duration_ms)],
    'energy':[float(energy)],
    'explicit':[int(explicit)],
    'id':[id],
    'instrumentalness':[float(instrumentalness)],
    'key':[int(key)],
    'liveness':[float(liveness)],
    'loudness':[float(loudness)],
    'mode':[int(mode)],
    'name':[name],
    'release_date':[release_date],
    'speechiness':[float(speechiness)],
    'tempo':[float(tempo)],
    'valence':[float(valence)],
    'artist':[artist],
    }
    X_pred=pd.DataFrame.from_dict(x_pred)


    #     acousticness        float64
    #     danceability        float64
    #     duration_ms           int64
    #     energy              float64
    #     explicit              int64
    #     id                   object
    #     instrumentalness    float64
    #     key                   int64
    #     liveness            float64
    #     loudness            float64
    #     mode                  int64
    #     name                 object
    #     release_date         object
    #     speechiness         float64
    #     tempo               float64
    #     valence             float64
    #     artist               object

    pipeline = joblib.load('model.joblib')
    prediction=pipeline.predict(X_pred)[0]

    return {"artist": artist,
    "name":name,
    "popularity":prediction}


    #/predict?acousticness=0.654&danceability=0.499&duration_ms=219827&energy=0.19&explicit=0&id=0B6BeEUd6UwFlbsHMQKjob&instrumentalness=0.00409&key=7&liveness=0.0898&loudness=-16.435&mode=1&name=Back%20in%20the%20Goodle%20Days&release_date=1971&speechiness=0.0454&tempo=149.46&valence=0.43&artist=John%20Hartford
