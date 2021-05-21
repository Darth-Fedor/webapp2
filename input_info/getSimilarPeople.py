import pandas as pd
import os

def getPeople(age, gender, race, sexorien, genid):
    
    SITE_ROOT = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_excel (SITE_ROOT + r'\Dataset.xlsx')
    for index, row in df.iterrows():
        if(row['Race'] != race or row['Gender'] != gender or row['Sexual_orientation'] != sexorien or row['Gender_identity'] != genid or abs(int(row['Age']) - int(age))>10):
            df.drop(index, inplace=True)
    return df
