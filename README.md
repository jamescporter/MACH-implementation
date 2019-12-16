# MACH-implementation

Working experiment with merged-average classification via hashing as outlined in this [paper](#https://arxiv.org/pdf/1910.13830.pdf) by Medini, et al.  Was presented at NeurIPS 2019 and I found it extremely interested.  

What exists right now is a simple quick mock-up with many limitations.  Right now, all models require a sklearn-esque wrapper that can be trained with the fit() method and predictions given using a predict() method.  Additionally, the categories of the target feature must be integer encoded starting as 0 through n-1, where n is the total number of categories to be predicted.  This means that for Tensorflow models, sparse_categorical_crossentropy should be used to calculate loss.

This is a bit of a generalization of the method presented in the paper *Extreme Classification in Log Memory using Count-Min Sketch: A Case Study of Amazon Search with 50M Products*.  In this case, any classification model can be used to train for each hash function and an additional classification model from the bucketed output of each model to the overall classification.
