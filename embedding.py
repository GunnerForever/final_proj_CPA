# Theoretically, I think we should split the model into half, each half being embeddings and decoding respectively, to make the "swap out perturbations" testing simpler
    #Embedding.py does the encoder half of the model, decoder is the decoding
# Alternatively we just make one CPA model file