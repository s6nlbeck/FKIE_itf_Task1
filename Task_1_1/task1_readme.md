#Subtask 1


The first subtask is about detecting events about riots and social movements in news articles. A single training instance looks like this:

> {"id":90000,"text":"   20\/02\/2004  - 18h33   A Semana: crise no Haiti se agrava, e Irã vai às urnas   As imagens desta se","label":0}



## Training

To train a model for subtask 1, we first need to transform the content of the file to a certain format. An example is given in the ```run_example()``` method.

For transforming the train data, we use the function ```create_doc_embeddings_list(path_in, path_out)```. This function read the specified file in, generates the document embeddings and write it to pickle file.

Now, there should be a pickle file, containing the transformed training data. This file needs to be read in. This is done with the function ```read_pickle(path_in)```.

Since we use a multilingual model, we may have multiple input files for the different languages. Whilst training, we need to train them on each language. Therefore, we read in all pickle files and concatenate them. To do so,
we can use the numpy concatenate function. Note, that we have to concatenate the embeddings and the labels separately. If this is done, we make a tuple of all embeddings and the labels. It should look kinda like this:
```
    all_data_train = np.concatenate([train_data_en[0], train_data_es[0], train_data_pr[0]], axis=0)
    all_label_train = np.concatenate([train_data_en[1], train_data_es[1], train_data_pr[1]], axis=0)
    data_train = (all_data_train, all_label_train)
```

Now, we can create all the nets. A call of the constructor of the ````Net_Container```` class is enough. It may look like this:
````
model = Net_Container(num_nets=10)
````

For training all these nets, we just call the `````fit````` method like this:
````
model.fit(data_train,epochs=20, stats=False)
````

The first parameter contains the tuple with the embeddings and the matching labels. The second parameter is the number of epochs.
The last parameter plots for every single net the train statistics if set True. To save the trained model, we need to create an empty folder and call the ````save```` method like this:
````
model.save("Example_data/Example_Model")
````

## Predicting

When we already have trained model, we need to load the model.