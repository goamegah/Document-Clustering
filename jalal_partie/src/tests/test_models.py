import pandas as pd
from src.core.processing.Preprocessing import Preprocessing

if __name__ =="__main__":
    file_name="/home/khaldi/Downloads/classic3.csv"
    df=pd.read_csv(file_name)
    df=df.drop(["Unnamed: 0"],axis=1)
    p=Preprocessing()
    df["tokens"]=df["text"].apply(lambda text:p.pipeline(text))
    print()