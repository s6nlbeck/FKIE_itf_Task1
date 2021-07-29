# Subtask 3

The third subtask is about event co-references in sentences. This means,
we have multiple sentences belonging to an event. The task is to decide which sentences
are about the same event and which belong to another event. For example, we may have an
instance like this:

````
{"event_clusters":[[4,5],[3]],"sentence_no":[3,4,5],"sentences":[
"The Tamil Nadu Toddy Movement , which comprises about 300 farmer ’ s associations in the State , launched a law breaking protest on January 21 , 2009 demanding the ban on toddy lifted.",
"On the same day TNCC President Thangkabalu led a fast in Chennai in favor of total prohibition including toddy .",
"The counter protest by the TNCC chief raised a furore with the Movement with its coordinator S Nallusamy saying Thangkabalu “ had insulted the feelings of farmers , coconut and Palmyra tree climbers consumers and nature lovers . ” A few days back Thangkabalu told reporters in Salem that he had decided on the date for his protest three months ago and did not intend it to be a counter protest to that started by the toddy farmers ."],"id":55194}

````

## Training
An example for the training is given in the main method.
To train a model you have to transform the data using the ```transfrom_to_sentence_embds``` method. This method 
creates for each instance the embeddings and combines them to positive and negative triples. With this transformed data,
the model can be trained as usual, using the fit function.

## Predicting

To predict the clusters, use the ```predict``` function. The predict function needs a list containing the instances to predict.
This means it takes as input a list of instances like:
```
[
(id, [(embd, sent_no)(.,..)(.,..)(.,..)(.,..)]),
(id, [(embd, sent_no)(.,..)(.,..)(.,..)(.,..)]),
]
```

To reproduce the results from the paper use the ```generate_submission``` function. This code
was used to produce the results of the paper.
