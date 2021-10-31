from utils.model import Perceptron
from utils.all_utills import prepare_data, save_model,save_plot
import pandas as pd
import numpy as np

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model_and = Perceptron(eta=ETA, epochs=EPOCHS)
model_and.fit(X, y)


save_model(model_and,filename="and.model")
save_plot(df, "and.png",model_and)