import fasttext
import pickle

def sentiment(text):
    if len(text):
        # read fasttext model and tfidf tranformer
        loaded_model = fasttext.load_model('model/fasttext_model')
            
        predict = loaded_model.predict(text)
        label = {'predicted label': predict[0][0],'Probability': predict[1][0]}
        return label
    return None