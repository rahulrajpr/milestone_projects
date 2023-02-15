# Getting the input

def prediction(news):
    # libraries required
    import numpy as np
    import pickle
    from tensorflow import keras as kr

    # getting the picke files for etl & prediction
    wcount = pickle.load(open('word_counter.pkl', 'rb'))
    token = pickle.load(open('tokenizer.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    classifier = pickle.load(open('classifier.pkl', 'rb'))

    # peforming the etl & prediction
    inp_array = np.array(news).reshape(-1)
    input_token = token.texts_to_sequences(inp_array)  # Tokenize the news headline string
    # Padding the tokenized varible - with maximum word count
    input_token_padded = kr.preprocessing.sequence.pad_sequences(input_token, maxlen=wcount, padding='post')
    input_token_padded_sc = scaler.transform(input_token_padded)  # scaling

    pred = classifier.predict(input_token_padded_sc)  # prediction
    pred_value = pred[0]
    return pred_value
