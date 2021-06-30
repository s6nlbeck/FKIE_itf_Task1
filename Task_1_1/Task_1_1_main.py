import json
import os
import flair
import torch
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from keras import layers
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import pickle
import numpy as np
import keras
import gc
import matplotlib.pyplot as plt


def create_model():
    """
    Creates a single neural net.
    :return: A Keras Model.
    """
    input_shape = (768,)
    input = keras.Input(shape=input_shape)
    x = layers.Dense(64, activation="sigmoid")(input)
    x = layers.Dense(64, activation="sigmoid")(x)
    x = layers.Dense(64, activation="sigmoid")(x)
    x = layers.Dense(64, activation="sigmoid")(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(input, out)
    model.summary(line_length=200)
    return model


def load_model(path):
    """
    Loads a single model.
    :param path: Path to the model
    :return: an object of class Task_1_Model, containing a Keras model.
    """
    model = keras.models.load_model(path)
    return Task_1_Model(model)


class Net_Container():
    """
    This class works as a container class. It is used to manage an arbitrary amount of neural nets.
    The methods do the same as on a single net, but in this case for all.
    """
    def __init__(self, num_nets=99, list_of_nets=None):
        """
        Creating a container without nets, the given number of nets is generated and stored in a list.
        The list also called nets list in the following, contains all the independent nets which are used for
        classification.
        :param num_nets: The number of nets that should be generated
        :param list_of_nets: Used if we already have trained nets. We can pass them as a list of objects of type
        Task_1_Model.
        """
        if list_of_nets is None:
            """
            The case, when we do not use pretrained nets.
            """
            self.nets = []
            for x in range(num_nets):
                """
                Generate the given amount of nets and store them in the list.
                """
                self.nets.append(Task_1_Model())

        else:
            """
            This is the case, when we have pretrained nets. There are no new generated.
            """
            self.nets = list_of_nets

    def fit(self, data, epochs=20, stats=True):
        """
        Uses the fit function with the given parameters for every single network in our model.
        :param data: The training data with embeddings and labels.
        :param epochs: Number of epochs we should train on.
        :param stats: Used if statistics should be shown after the training
        :return: Nothing
        """
        for net in self.nets:
            """
            Train the nets without saving them.
            """
            net.fit_wo_save(data, epochs=epochs, stats=stats)

    def save(self, path):
        """
        Saves all models in a folder. The path has to be a folder.
        :param path: Folder where the nets are saved.
        :return: Nothing
        """
        counter = 0
        for net in self.nets:
            print("Save network: ", counter)
            counter += 1
            net.save_model(path + "/model" + str(counter))

    def load_nets(self, path):
        """
        Load all the nets in the given folder. The folder should only contain saved nets, nothing else.
        :param path: Path to the folder containing the trained nets
        :return: Nothing
        """
        loaded_nets = []
        files = os.listdir(path)

        """
        Iterate over every entry in the directory and load each model.
        """
        counter = 0
        for file in files:
            print("Load network: ", counter)
            counter += 1
            ln = load_model(path + "/" + file)
            loaded_nets.append(ln)
        self.nets = loaded_nets

    def predict(self, instance):
        """
        Used to predict the label for one instance.
        :param instance: A document embedding, where the label should be predicted.
        :return: the label as integer.
        """
        sum_preds = 0
        """
        Get the prediction for each net and sum them up.
        """
        for net in self.nets:
            sum_preds += net.predict(instance)
        """
        Calculating the average of the responses of the nets is used to calculate the majority vote.
        """
        sum_preds /= len(self.nets)
        if sum_preds >= 0.5:
            return 1
        else:
            return 0

    def predict_raw(self, data):
        """
        Used to predict the raw label for one instance. Returns the calculated average.
        :param data: A document embedding.
        :return: The avaraged prediction of the nets.
        """
        sum_preds = 0
        for net in self.nets:
            sum_preds += net.predict(data)
        sum_preds /= len(self.nets)
        return sum_preds


class Task_1_Model():
    def __init__(self, model=None):
        """
        This class contains a Keras model and some useful functions. If we have a trained net we can use this net
        as parameter. Otherwise a new neural network is created.
        :param model:
        """
        if model is not None:
            self.model = model
        else:
            self.model = create_model()
            model_loss = keras.losses.BinaryCrossentropy()
            optim = keras.optimizers.Adam(learning_rate=0.001)
            self.model.compile(optimizer=optim, loss=model_loss, metrics=['acc', 'mse'])

    def save_model(self, path):
        """
        Saves this model to disk.
        :param path: Path to the place where it should be stored.
        :return: Nothing
        """
        self.model.save(path)

    def fit_wo_save(self, data, epochs=20, stats=False):
        """
        Fits the model to the given data. The data is expected to be a tuple containing a Tensor with the embeddings and
        a tensor containing the labels.
        :param data: Tuple with the training instances and labels
        :param epochs: Number of epochs we train
        :param stats: If true, after the training, a plot with the training statistics is plotted.
        :return: nothing
        """
        examples = data[0]
        labels = data[1]
        history = self.model.fit(examples, labels, epochs=epochs, validation_split=0.2, batch_size=32)

        if stats:
            plt.plot(history.history["loss"], label="Loss train")
            plt.plot(history.history["val_loss"], label="Loss validation")
            plt.plot(history.history["val_acc"], label="Accuracy validation")
            plt.plot(history.history["acc"], label="Accuracy train")
            plt.plot(history.history["val_mse"], label="MSE validation")
            plt.plot(history.history["mse"], label="MSE train")
            plt.legend()
            plt.show()

    def fit(self, data):
        """
        Is not used. Can quickly fit and save a model.
        :param data: training data
        :return: nothing
        """
        examples = data[0]
        labels = data[1]

        history = self.model.fit(examples, labels, epochs=20, validation_split=0.2, batch_size=32)
        self.model.save("model_doc")
        plt.plot(history.history["loss"], label="Loss train")
        plt.plot(history.history["val_loss"], label="Loss validation")
        plt.plot(history.history["val_acc"], label="Accuracy validation")
        plt.plot(history.history["acc"], label="Accuracy train")
        plt.plot(history.history["val_mse"], label="MSE validation")
        plt.plot(history.history["mse"], label="MSE train")
        plt.legend()
        plt.show()

    def predict(self, data):
        """
        Given a document embedding, this functions gives the output of the trained net.
        :param data: An instance. Document embedding.
        :return: The predicted label as integer.
        """
        res = self.model.predict(np.array([data]))
        if res >= 0.5:
            return 1
        else:
            return 0


"""
Daten haben das Format [list with embeddings, list with labels ]
"""


def read_pickle(path):
    """
    Reads in a pickle file with word embeddings. Should be an Array with embeddings.
    Returns an list with the embeddings as torch tensor.
    :param path: path to the input file
    :return: the read pickle file
    """
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data


def create_doc_embeddings_list_test_data(data, path_out):
    """
    This method is used for transforming the given test dataset of subtask 1 to a form, which can be used as input for a
    trained model. Is used when there are no labels.
    :param data: the data which should be transformed.
    :param path_out: Path of the output file
    :return: the transformed data
    """
    instances = []
    """
    Using CPU for torch prevents out of memory for the graphics card. It's still fast enough.
    """
    flair.device = torch.device('cpu')
    """
    Choose a cased multilingual Bert transformer embedding
    """
    embedding = TransformerDocumentEmbeddings('bert-base-multilingual-cased')

    counter = 0
    """
    Iterate over the complete dataset and generate the embeddings.
    """
    for entry in data:
        print("Instance: ", counter)
        text = entry["text"]
        id = entry["id"]
        """
        Use the flair Toolkit for generating the embedding
        """
        flair_sentence = Sentence(text)
        embedding.embed(flair_sentence)
        """
        Run Garbage collector to prevent out of memory
        """
        gc.collect()
        vec = flair_sentence.get_embedding().detach().numpy()
        counter += 1
        """
        The embedding builds a tuple with the id.
        """
        tpl = (id, vec)
        instances.append(tpl)

    """
    Write the list to a file such that the data can be reused if we have different models to test.
    """
    file = open(path_out, "wb")
    pickle.dump(instances, file)
    file.close()
    """
    Return the list with the tuples. The list contains the tuples (id, embedding). So for each example we have now
    a tuple, representing the example.
    """
    return instances


def create_doc_embeddings_list(path_in, path_out):
    """
    This function is used to transform the training data. The input and output of this function are paths.
    The file at the given path is read in and transformed to a tuple containing two numpy arrays.
    The first element of the Tuple contains a two dimensional matrix with the generated document embeddings.
    The second one is a vector, containing the labels for each embedding.
    :param path_in: path to the training file
    :param path_out: path for the generated output file.
    :return: nothing
    """
    embeddings = []
    labels = []

    flair.device = torch.device('cpu')
    data = read(path_in)
    """
    Use a multilingual Bert model for generating sentence embeddings.
    """
    embedding = TransformerDocumentEmbeddings('bert-base-multilingual-cased')
    counter = 0

    print("Number of instances in file: ", len(data))

    for entry in data:
        print("Processing instance: ", counter)
        """
        The text variable hold the text for the given instance.
        the label_instance contains the matching label for that very instance.
        """
        text = entry["text"]
        label_instance = entry["label"]
        """
        Using flair to transform the text to a 768 dimensional embedding.
        """
        flair_sentence = Sentence(text)
        embedding.embed(flair_sentence)
        document_embedding = flair_sentence.get_embedding().detach().numpy()
        """
        Add the embedding to the list of embeddings and the label to the list with labels.
        """
        embeddings.append(document_embedding)
        labels.append(np.array([label_instance]))
        gc.collect()
        counter += 1
    """
    Transform both lists to a numpy array and write them to disk.
    """
    result = (np.array(embeddings), np.array(labels))
    print("Shape data: ", result[0].shape)
    print("Shape label: ", result[1].shape)
    file = open(path_out, "wb")
    pickle.dump(result, file)
    file.close()


def read(path):
    """
    Reads the file from the given path (json file).
    The file shall contain one json instance per row.
    The file is read in row by row.
    Returns list of instance dictionaries.
    :param path: The path to the json train file.
    :return: A list containing all instances in the file.
    """
    data = []
    file = open(path, "r", encoding="utf8")
    """
    Read the given file line for line and append the deserialized object to the list. 
    """
    for instance in file:
        data.append(json.loads(instance))
    file.close()
    return data


def evaluate_on_embeddings(model, data, path_result):
    labels, preds = [], []
    size_of_dataset = len(data[0])
    counter = 0

    for instance in range(len(data[0])):
        print("Fortschritt: ", str((counter / size_of_dataset) * 100) + "%")
        counter += 1
        gt = data[1][instance].item()
        labels.append(gt)
        res = model.predict(data[0][instance])
        print("Ground truth: {}, Predicted: {}".format(gt, res))
        preds.append(res)

    with open(path_result, 'w') as f:
        f.write(classification_report(labels, preds))
        print(classification_report(labels, preds))
    f1 = f1_score(labels, preds, average='macro')
    f1_1 = f1_score(labels, preds, average='weighted')

    print("F1 score macro: ", f1)
    print("F1 score weighted: ", f1_1)

    with open(path_result, 'w') as f:
        f.write(classification_report(labels, preds))
        print(classification_report(labels, preds))
    f1 = f1_score(labels, preds, average='macro')
    f1_1 = f1_score(labels, preds, average='weighted')

    print("F1 score macro: ", f1)
    print("F1 score weighted: ", f1_1)


def write_pickle(path, obj):
    """
    Writes an given object to a pickle file.
    :param path: Path where the object should be saved
    :param obj: The object which should be serialized.
    :return: Nothing.
    """
    fp = open(path, "wb")
    pickle.dump(obj, fp)
    fp.close()


def split_data():
    data_en = read_pickle("Sent_Embds/en-sent.pkl")
    data_es = read_pickle("Sent_Embds/es-sent.pkl")
    data_pr = read_pickle("Sent_Embds/pr-sent.pkl")

    data_en_d = data_en[0]
    dada_en_l = data_en[1]

    data_es_d = data_es[0]
    dada_es_l = data_es[1]

    data_pr_d = data_pr[0]
    dada_pr_l = data_pr[1]

    split_en = int(len(data_en_d) * 0.8)
    split_es = int(len(data_es_d) * 0.8)
    split_pr = int(len(data_pr_d) * 0.8)

    en_train = (data_en_d[:split_en], dada_en_l[:split_en])
    es_train = (data_es_d[:split_es], dada_es_l[:split_es])
    pr_train = (data_pr_d[:split_pr], dada_pr_l[:split_pr])

    en_test = (data_en_d[split_en:], dada_en_l[split_en:])
    es_test = (data_es_d[split_es:], dada_es_l[split_es:])
    pr_test = (data_pr_d[split_pr:], dada_pr_l[split_pr:])

    write_pickle("en-train.pkl", en_train)
    write_pickle("es-train.pkl", es_train)
    write_pickle("pr-train.pkl", pr_train)

    write_pickle("en-test.pkl", en_test)
    write_pickle("es-test.pkl", es_test)
    write_pickle("pr-test.pkl", pr_test)


def generate_submission(transformed_data, model, path):
    result_list = []
    counter = 0
    lenght = len(transformed_data)
    for instance in transformed_data:
        print("Progress: ", (counter / lenght * 100))
        counter += 1
        id = instance[0]
        vec = instance[1]
        res = model.predict(vec)
        result_list.append({"id": id, "prediction": res})
    with open(path + ".json", "w", encoding="utf-8") as f:
        for doc in result_list:
            f.write(json.dumps(doc) + "\n")


def submission_routine():
    """
    We do all the submission routine. Read in the test set, transform it, load the model, predict the values
    and write it all to a file.
    :return: Nothing
    """

    """
    Read in the files with the test data. They contain no labels.
    """
    t_en = read("Data_Test/test-en.json")
    t_es = read("Data_Test/test-es.json")
    t_pr = read("Data_Test/test-pr.json")
    t_hi = read("Data_Test/test-hi.json")

    """
    Generate the embeddings.
    """
    transformed_data_en = create_doc_embeddings_list_test_data(t_en, "Data_Test/embd-en.pkl")
    transformed_data_es = create_doc_embeddings_list_test_data(t_es, "Data_Test/embd-es.pkl")
    transformed_data_pr = create_doc_embeddings_list_test_data(t_pr, "Data_Test/embd-pr.pkl")
    transformed_data_hi = create_doc_embeddings_list_test_data(t_hi, "Data_Test/embd-hi.pkl")

    """
    Read in the generated embeddings.
    """
    transformed_data_en = read_pickle("Data_Test/embd-en.pkl")
    transformed_data_es = read_pickle("Data_Test/embd-es.pkl")
    transformed_data_pr = read_pickle("Data_Test/embd-pr.pkl")
    transformed_data_hi = read_pickle("Data_Test/embd-hi.pkl")

    """
    Load the model
    """
    model = Net_Container(list_of_nets=[])
    model.load_nets("Net_container_nets_trained_on2021")
    model.nets[0].model.summary()


    generate_submission(transformed_data_en, model, "paper-one-net-submission-en")
    generate_submission(transformed_data_es, model, "paper-one-net-submission-es")
    generate_submission(transformed_data_pr, model, "paper-one-net-submission-pr")
    generate_submission(transformed_data_hi, model, "paper-one-net-submission-hi")

def run_example():
    """
    This is an example for the main part. It uses all functions to train and evaluate the network.
    :return: Nothing
    """
    train = False
    if train:
        """
        First, we read in the training data file.    
        """
        create_doc_embeddings_list("Example_data/en-train.json", "Example_data/en-train-embeddings.pkl")
        create_doc_embeddings_list("Example_data/es-train.json", "Example_data/es-train-embeddings.pkl")
        create_doc_embeddings_list("Example_data/pr-train.json", "Example_data/pr-train-embeddings.pkl")

        """
        Read in the embeddings.
        """
        train_data_en = read_pickle("Example_data/en-train-embeddings.pkl")
        train_data_es = read_pickle("Example_data/es-train-embeddings.pkl")
        train_data_pr = read_pickle("Example_data/pr-train-embeddings.pkl")

        """
        Build a multilingual train set by concatenating the different sets.
        """
        all_data_train = np.concatenate([train_data_en[0], train_data_es[0], train_data_pr[0]], axis=0)
        all_label_train = np.concatenate([train_data_en[1], train_data_es[1], train_data_pr[1]], axis=0)
        data_train = (all_data_train, all_label_train)

        model = Net_Container(num_nets=10)

        model.fit(data_train,epochs=20, stats=False)
        model.save("Example_data/Example_Model")

    else:

        model = Net_Container(list_of_nets=[])
        model.load_nets("Example_data/Example_Model")

        test_data = read("Example_data/test-en.json")
        transformed_test_data = create_doc_embeddings_list_test_data(test_data,"Example_data/test-data-en.pkl")

        generate_submission(transformed_test_data, model, "Example_data/predictions")

        print(model.predict(transformed_test_data[0][1]))




if __name__ == "__main__":
    # Makes Submsions
    # submission_routine()
    run_example()
    exit()

    train_mode = True
    data_en = read_pickle("Sent_Embds/en-sent.pkl")
    data_es = read_pickle("Sent_Embds/es-sent.pkl")
    data_pr = read_pickle("Sent_Embds/pr-sent.pkl")

    all_data_train = np.concatenate([data_en[0], data_es[0], data_pr[0]], axis=0)
    all_label_train = np.concatenate([data_en[1], data_es[1], data_pr[1]], axis=0)

    if train_mode == False:
        data_en = read_pickle("Sent_Embds/en-test.pkl")
        data_es = read_pickle("Sent_Embds/es-test.pkl")
        data_pr = read_pickle("Sent_Embds/pr-test.pkl")

        all_data_train = np.concatenate([data_en[0], data_es[0], data_pr[0]], axis=0)
        all_label_train = np.concatenate([data_en[1], data_es[1], data_pr[1]], axis=0)
        model = Net_Container(list_of_nets=[])
        model.load_nets("Net_container_nets_trained_on2021")
        evaluate_on_embeddings(model, (all_data_train, all_label_train), "result_multi_net")


    else:
        data_en = read_pickle("Sent_Embds/en-train.pkl")
        data_es = read_pickle("Sent_Embds/es-train.pkl")
        data_pr = read_pickle("Sent_Embds/pr-train.pkl")

        all_data_train = np.concatenate([data_en[0], data_es[0], data_pr[0]], axis=0)
        all_label_train = np.concatenate([data_en[1], data_es[1], data_pr[1]], axis=0)
        # data_to_train = """unbias_classes"""((all_data_train,all_label_train))
        data_to_train = (all_data_train, all_label_train)

        # model = Net_Container()
        model = Net_Container(num_nets=100)
        model.fit(data_to_train, epochs=20, stats=False)
        model.save("trained_model")
