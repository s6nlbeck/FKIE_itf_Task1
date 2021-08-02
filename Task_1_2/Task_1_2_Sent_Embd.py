import gc
import json
import pickle
import flair
import torch
from sklearn.metrics import classification_report, f1_score
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt



def create_model():
    """
    Creates a Keras model as in the paper.
    :return:
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
    model = keras.models.load_model(path)
    return Model_1_2_Sent(model)


class Model_1_2_Sent:
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = create_model()

            model_loss = keras.losses.BinaryCrossentropy()
            optim = keras.optimizers.Adam(learning_rate=0.001)
            self.model.compile(optimizer=optim, loss=model_loss, metrics=['acc', 'mse'])

    def save_model(self, path):
        self.model.save(path)

    def fit_wo_save(self, data):
        """
        Learns the seed for future prediction.
        Doesnt use the training file.
        """
        examples = data[0]
        labels = data[1]
        self.model.fit(examples, labels, epochs=20, validation_split=0.2, batch_size=32)

    def fit(self, data):
        """
        Learns the seed for future prediction.
        Doesnt use the training file.
        """
        examples = data[0]
        labels = data[1]

        history = self.model.fit(examples, labels, epochs=5, validation_split=0.2, batch_size=32)
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
        if res >= 0.5:
            return 1
        else:
            return 0


def read_json_file(path):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))
    return data

def generate_embeddings_sentence_test_data(data, path_out):
    """
    Generates the embeddings for unlabeled data.
    :param data: List with instances
    :param path_out: output file
    :return: list with embeddings
    """
    flair.device = torch.device('cpu')
    dicts = []
    # init multilingual BERT
    bert_embedding = TransformerDocumentEmbeddings('bert-base-multilingual-cased')
    counter = 0
    for entry in data:
        print("Counter: ", counter)
        counter += 1
        text = entry["sentence"]
        id = entry["id"]
        sent = Sentence(text)
        bert_embedding.embed(sent)
        vec = sent.get_embedding().detach().numpy()
        dicts.append((id,vec))
        gc.collect()
    result = dicts
    file = open(path_out, "wb")
    pickle.dump(result, file)
    file.close()
    return result



def generate_embeddings_sentence(path_in, path_out):
    list_of_dicts = read_json_file(path_in)
    flair.device = torch.device('cpu')

    emds = []
    labels = []

    # init multilingual BERT
    bert_embedding = TransformerDocumentEmbeddings('bert-base-multilingual-cased')

    counter = 0
    for entry in list_of_dicts:
        print("Counter: ", counter)
        counter += 1
        text = entry["sentence"]
        label = entry["label"]
        sent = Sentence(text)
        bert_embedding.embed(sent)
        vec = sent.get_embedding().detach().numpy()
        print("Shape of emnd: ", vec.shape)
        gc.collect()
        emds.append(vec)
        labels.append(label)
    result = (emds, labels)
    file = open(path_out, "wb")
    pickle.dump(result, file)
    file.close()


def evaluate_on_embeddings(model, data, path_result):
    labels, preds = [], []
    for instance in range(len(data[0])):
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


def read_gen_file(path_in):
    print("Read file")
    fp = open(path_in, "rb")
    data = pickle.load(fp)
    fp.close()
    print("Read file completed")
    return data[0], data[1]


def read_pickle(path_in):
    print("Read file")
    fp = open(path_in, "rb")
    data = pickle.load(fp)
    fp.close()
    print("Read file completed")
    return data

def make_submission(data, model, path):
    """
    Function to generate the submission as requested for codalab
    :param data:
    :return:
    """
    counter = 0
    length= len(data)
    test_predictions = []
    #Data has form of [(id,vec),(id,vec)....]
    for instance in data:
        print("Prog: ",(counter/length*100))
        counter+=1
        id = instance[0]
        vec = instance[1]
        res = model.predict(vec)
        print("Predicted: ",res)
        test_predictions.append({"id":id,"prediction":res})
    with open(path+".json", "w", encoding="utf-8") as f:
        for doc in test_predictions:
            f.write(json.dumps(doc) + "\n")


def gen_train_test_sets(path_in, path_out_train, path_out_test):
    """
    Method to split the train data in .8 : .2 ratio for having a train and test set for development
    :param path_in: path of input file
    :param path_out_train: path of output train file
    :param path_out_test: path of output test file
    :return: Darkness
    """
    data, label = read_gen_file(path_in)
    split_index = int(len(data) * 0.8)
    data_train = data[:split_index]
    label_train = label[:split_index]
    data_test = data[split_index:]
    label_test = label[split_index:]
    train_tpl = (data_train, label_train)
    test_tpl = (data_test, label_test)
    fp = open(path_out_train, "wb")
    pickle.dump(train_tpl, fp)
    fp.close()
    fp2 = open(path_out_test, "wb")
    pickle.dump(test_tpl, fp2)
    fp2.close()


def generate_submissons_all_steps():
    """
    Makes the submission for codalab
    :return:
    """


    data_en = read_json_file("Test_Data/test-en.json")
    data_pr = read_json_file("Test_Data/test-pr.json")
    data_es = read_json_file("Test_Data/test-es.json")
    res_en = generate_embeddings_sentence_test_data(data_en, "Test_Data/embd-en.pkl")
    res_es = generate_embeddings_sentence_test_data(data_es, "Test_Data/embd-es.pkl")
    res_pr = generate_embeddings_sentence_test_data(data_pr, "Test_Data/embd-pr.pkl")
    model = load_model("model_doc")
    make_submission(res_es, model, "submission-es")
    make_submission(res_pr, model, "submission-pr")
    make_submission(res_en, model, "submission-en")
    exit()


def generate_sentence_embeddings():
    """
    Does what the name says
    :return:
    """
    generate_embeddings_sentence("Data/en-train.json", "Data_Sent_Embds/en_sent.pkl")
    generate_embeddings_sentence("Data/es-train.json", "Data_Sent_Embds/es_sent.pkl")
    generate_embeddings_sentence("Data/pr-train.json", "Data_Sent_Embds/pr_sent.pkl")



def read_and_split_sets():
    """
    Reads in the embedded sets and split them up in train and test sets
    :return:
    """
    gen_train_test_sets("Data_Sent_Embds/en_sent.pkl", "Data_Sent_Embd_Splitted/en_train.pkl",
                        "Data_Sent_Embd_Splitted/en_test.pkl")
    gen_train_test_sets("Data_Sent_Embds/es_sent.pkl", "Data_Sent_Embd_Splitted/es_train.pkl",
                        "Data_Sent_Embd_Splitted/es_test.pkl")
    gen_train_test_sets("Data_Sent_Embds/pr_sent.pkl", "Data_Sent_Embd_Splitted/pr_train.pkl",
                        "Data_Sent_Embd_Splitted/pr_test.pkl")



if __name__ == "__main__":
    #This is used for makeing the submission with the given model
    #generate_submissons_all_steps()


    # Generates the embeddings
    #generate_sentence_embeddings()


    # Reads in Data and creates train and test set
    #read_and_split_sets()

    train = True



    if train == False:
        model = load_model("model_doc")
        en_train_tpl = read_pickle("Data_Sent_Embd_Splitted/en_test.pkl")
        es_train_tpl = read_pickle("Data_Sent_Embd_Splitted/es_test.pkl")
        pr_train_tpl = read_pickle("Data_Sent_Embd_Splitted/pr_test.pkl")
        all_data_train = np.concatenate([en_train_tpl[0], es_train_tpl[0], pr_train_tpl[0]], axis=0)
        all_label_train = np.concatenate([en_train_tpl[1], es_train_tpl[1], pr_train_tpl[1]], axis=0)
        all_tpl_train = (all_data_train, all_label_train)
        evaluate_on_embeddings(model, all_tpl_train, "res_1_2_sent.txt")
    else:
        en_train_tpl = read_pickle("Data_Sent_Embd_Splitted/en_train.pkl")
        es_train_tpl = read_pickle("Data_Sent_Embd_Splitted/es_train.pkl")
        pr_train_tpl = read_pickle("Data_Sent_Embd_Splitted/pr_train.pkl")
        all_data_train = np.concatenate([en_train_tpl[0], es_train_tpl[0], pr_train_tpl[0]], axis=0)
        all_label_train = np.concatenate([en_train_tpl[1], es_train_tpl[1], pr_train_tpl[1]], axis=0)
        all_tpl_train = (all_data_train, all_label_train)
        model = Model_1_2_Sent()
        model.fit(all_tpl_train)
