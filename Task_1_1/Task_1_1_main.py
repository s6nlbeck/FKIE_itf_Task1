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
    model = keras.models.load_model(path)
    return Task_1_Model(model)


class Net_Container():
    def __init__(self,num_nets=99, list_of_nets=None):
        if list_of_nets is None:
            self.nets=[]
            for x in range(num_nets):
                self.nets.append(Task_1_Model())

        else:
            self.nets = list_of_nets

    def fit(self, data, epochs=20, stats = True):
        for net in self.nets:
            net.fit_wo_save(data, epochs = epochs, stats = stats)
            #net.fit(data)
    def save(self, path):
        counter = 0
        for net in self.nets:
            print("Save net: ",counter)
            counter+=1
            net.save_model(path+"/model"+str(counter))

    def load_nets(self, path):
        loaded_nets=[]
        files = os.listdir(path)
        counter = 0
        for file in files:
            print("Load net: ",counter)
            counter += 1
            ln = load_model(path+"/"+file)
            loaded_nets.append(ln)
        self.nets = loaded_nets



    def predict(self, data):
        sum_preds=0
        for net in self.nets:
            sum_preds+=net.predict(data)
        sum_preds/=len(self.nets)
        if sum_preds>=0.5:
            return 1
        else:
            return 0

    def predict_raw(self, data):
        sum_preds=0
        for net in self.nets:
            sum_preds+=net.predict(data)
        sum_preds/=len(self.nets)
        return sum_preds


class Task_1_Model():
    def __init__(self, model =None):
        if model is not None:
            self.model = model
        else:
            self.model = create_model()

            model_loss = keras.losses.BinaryCrossentropy()
            optim =  keras.optimizers.Adam(learning_rate=0.001)
            self.model.compile(optimizer=optim, loss = model_loss, metrics=['acc','mse'])


    def save_model(self,path):
        self.model.save(path)

    def fit_wo_save(self, data, epochs = 20, stats = False):
        """
        Learns the seed for future prediction.
        Doesnt use the training file.
        """
        examples = data[0]
        labels = data[1]
        history = self.model.fit(examples, labels, epochs=epochs, validation_split=0.2,  batch_size=32)
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
        Learns the seed for future prediction.
        Doesnt use the training file.
        """
        examples = data[0]
        labels = data[1]


        history = self.model.fit(examples, labels, epochs=200, validation_split=0.2,  batch_size=32)
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
        res = self.model.predict(np.array([data]))
        if res>=0.5:
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
    """
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def create_doc_embds_list_test_data(data, path_out):
    instances = []
    flair.device = torch.device('cpu')

    embedding = TransformerDocumentEmbeddings('bert-base-multilingual-cased')
    counter = 0
    for entry in data:
        print("Counter: ",counter)
        text = entry["text"]
        id =  entry["id"]
        flair_sentence = Sentence(text)
        embedding.embed(flair_sentence)
        gc.collect()
        vec = flair_sentence.get_embedding().detach().numpy()
        counter += 1
        tpl = (id,vec)
        instances.append(tpl)


    file = open(path_out, "wb")
    pickle.dump(instances, file)
    file.close()
    return instances



def create_doc_embds_list(path_in, path_out):
    emds = []
    label = []
    flair.device = torch.device('cpu')

    data = read(path_in)

    print("LENGTH OF DATA: ",len(data))
    embedding = TransformerDocumentEmbeddings('bert-base-multilingual-cased')
    counter = 0
    for entry in data:
        print("Counter: ",counter)
        text = entry["text"]
        label_ins = entry["label"]
        flair_sentence = Sentence(text)
        embedding.embed(flair_sentence)
        npembd = flair_sentence.get_embedding().detach().numpy()
        print("Shape of Embedding: ", npembd.shape)
        gc.collect()
        emds.append(flair_sentence.get_embedding().detach().numpy())
        label.append(np.array([label_ins]))
        counter += 1
    result = (np.array(emds), np.array(label))
    print("Shape data: ", result[0].shape)
    print("Shape label: ", result[1].shape)
    file = open(path_out, "wb")
    pickle.dump(result, file)
    file.close()

def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    file = open(path, "r", encoding="utf8")
    for instance in file:
        print(instance)
        data.append(json.loads(instance))
    file.close()
    return data

def evaluate_on_embeddings(model, data, path_result):
    labels, preds = [], []
    size_of_dataset = len(data[0])
    counter=0

    for instance in range(len(data[0])):
        print("Fortschritt: ",str((counter/size_of_dataset)*100)+"%")
        counter+=1
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
    counter =0
    lenght = len(transformed_data)
    for instance in transformed_data:
        print("Progress: ",(counter/lenght*100))
        counter+=1
        id = instance[0]
        vec = instance[1]
        res = model.predict(vec)
        result_list.append({"id":id, "prediction":res})
    with open(path+".json", "w", encoding="utf-8") as f:
        for doc in result_list:
            f.write(json.dumps(doc) + "\n")


def submission_routine():
    """
    We do all the submission routine
    :return:
    """

    t_en = read("Data_Test/test-en.json")
    t_es = read("Data_Test/test-es.json")
    t_pr = read("Data_Test/test-pr.json")
    t_hi = read("Data_Test/test-hi.json")



    transformed_data_en = create_doc_embds_list_test_data(t_en,"Data_Test/embd-en.pkl")
    transformed_data_es = create_doc_embds_list_test_data(t_es, "Data_Test/embd-es.pkl")
    transformed_data_pr = create_doc_embds_list_test_data(t_pr, "Data_Test/embd-pr.pkl")
    transformed_data_hi = create_doc_embds_list_test_data(t_hi, "Data_Test/embd-hi.pkl")

    transformed_data_en = read_pickle("Data_Test/embd-en.pkl")
    transformed_data_es = read_pickle("Data_Test/embd-es.pkl")
    transformed_data_pr = read_pickle("Data_Test/embd-pr.pkl")
    transformed_data_hi = read_pickle("Data_Test/embd-hi.pkl")



    model = Net_Container(list_of_nets=[])
    model.load_nets("Net_container_nets_trained_on2021")
    model.nets=model.nets[:1]
    model.nets[0].model.summary()

    print("Num nets: ",len(model.nets))

    generate_submission(transformed_data_en, model,"paper-one-net-submission-en")
    generate_submission(transformed_data_es, model, "paper-one-net-submission-es")
    generate_submission(transformed_data_pr, model, "paper-one-net-submission-pr")
    generate_submission(transformed_data_hi, model, "paper-one-net-submission-hi")



if __name__ == "__main__":
    #Makes Submsions
    #submission_routine()

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
        evaluate_on_embeddings(model, (all_data_train,all_label_train), "result_multi_net")


    else:
        data_en = read_pickle("Sent_Embds/en-train.pkl")
        data_es = read_pickle("Sent_Embds/es-train.pkl")
        data_pr = read_pickle("Sent_Embds/pr-train.pkl")

        all_data_train = np.concatenate([data_en[0], data_es[0], data_pr[0]], axis=0)
        all_label_train = np.concatenate([data_en[1], data_es[1], data_pr[1]], axis=0)
        #data_to_train = """unbias_classes"""((all_data_train,all_label_train))
        data_to_train = (all_data_train,all_label_train)

        #model = Net_Container()
        model = Net_Container(num_nets=1)
        model.fit(data_to_train,epochs=100, stats = True)
        model.save("trained_model")


