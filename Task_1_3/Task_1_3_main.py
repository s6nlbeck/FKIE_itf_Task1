import json
import itertools
import argparse
import pickle
import subprocess
import networkx as nx
import flair
import keras
import torch
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from keras import layers
import numpy as np



flair.device = torch.device('cpu')
bert_embedding = TransformerDocumentEmbeddings('bert-base-multilingual-cased')

def calc_sent_embd(text):
    sent = Sentence(text)
    bert_embedding.embed(sent)
    vec = sent.get_embedding().detach().numpy()
    return vec

def get_no_sent(tuples_list, nummer):
    """
    This functions returns the sentence correspoding to the number.
    :param tuples_list:
    :param nummer:
    :return:
    """
    for tuple in tuples_list:
        if tuple[0] == nummer:
            return tuple[1]
    raise IOError()


def transfrom_to_sentence_embds(data,out_path):
    """
    Uses the training data and outputs a file consisting of data to feed to a neural net.
    In this method the sentences of each instance are combined to produce positive and negative examples.
    In this method also the embeddings are calculated.
    :param data: List with instances
    :param out_path: path of the output file
    :return: return the calculated list
    """

    result_list = []
    for instance in data:
        # Creates a list with number-sentence tuples
        tuples_mat_nr = list(zip(instance["sentence_no"], instance["sentences"]))
        clusters = instance["event_clusters"]
        embd_clusters = []
        # Run over all clusters
        for cl in clusters:
            this_cl_embd = []
            for sent_num in cl:
                sent = get_no_sent(tuples_mat_nr, sent_num)
                #Calculate embedding
                result = calc_sent_embd(sent)
                this_cl_embd.append(result)
            embd_clusters.append(this_cl_embd)
        # Now we should have [[emd1,emd2],[emd3,emd4,emd5]]
        # Now we have to combine them to lists with a label
        # Run over the embd clusters and put each to each
        for cl in embd_clusters:
            for set in itertools.combinations(cl, 2):
                first_e = set[0]
                second_e = set[1]
                res_tpl = (first_e, second_e, 1)
                result_list.append(res_tpl)
        # Now we have all positive combinations. Now do the negatives.
        if len(embd_clusters) > 1:
            for tuples in itertools.combinations(embd_clusters, 2):
                for comb in itertools.product(*tuples):
                    new_one = (comb[0], comb[1], 0)
                    result_list.append(new_one)
    file = open(out_path, "wb")
    pickle.dump(result_list, file)
    file.close()
    return result_list

class Comparer():
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = self.create_model()

            model_loss = keras.losses.BinaryCrossentropy()
            optim = keras.optimizers.Adam(learning_rate=0.001)
            self.model.compile(optimizer=optim, loss=model_loss, metrics=['acc', 'mse'])

    def create_model(self):
        input_shape = (768,)
        input = keras.Input(shape=input_shape)
        input2 = keras.Input(shape=input_shape)
        x1 = layers.Dense(128, activation="sigmoid")(input)
        x2 = layers.Dense(128, activation="sigmoid")(input2)
        x = layers.Concatenate()([x1, x2])
        x = layers.Dense(128, activation="sigmoid")(x)
        x = layers.Dense(64, activation="sigmoid")(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model([input,input2], out)
        model.summary(line_length=200)
        return model

    def predict_same_cluster(self, sent_1, sent_2, threshold =0.5):
        #Put the two sentences in the net to get the similarity score
        result_net = self.model.predict([np.array([sent_1]),np.array([sent_2])])
        #Check if it's above threshold 0, means they are not in a cluster 1, that they are.
        if result_net>=threshold:
            return True
        return False

    @classmethod
    def load(cls,path):
        model = keras.models.load_model(path)
        return cls(model)


    def save(self, path):
        self.model.save(path)

    def fit(self, data, epochs = 20):
        # [vec1,vec2,label]

        mat_1_list = []
        mat_2_list = []
        labels = []
        for example in data:
            mat1 = example[0]
            mat2 = example[1]
            lab = example[2]
            mat_1_list.append(mat1)
            mat_2_list.append(mat2)
            labels.append(lab)

        mat_1_list = np.array(mat_1_list)
        mat_2_list = np.array(mat_2_list)
        labels = np.array(labels)

        x = [mat_1_list,mat_2_list]
        y = labels
        self.model.fit(x, y, epochs=epochs, validation_split=0.2, batch_size=64)



def read_json(path):
    fp = open(path, "r")
    data = json.load(fp)
    fp.close()
    return data


def generate_test_data(data, path_out):
    """
    Transformes the data to a form such that predicting with the model is possible.
    :param data: list of the read in instances
    :param path_out: path of the file after transforming
    :return: The transformed data
    """
    list_with_isntances = []
    """The data has something of the form of tuples
    (id, [(embd, sent_no)(.,..)(.,..)(.,..)(.,..)]) organized as a list. """
    couinter = 0
    for thing in data:
        print(couinter)
        couinter+=1
        #Make a list for the tuples of num and sentence
        tpls = []
        id = thing["id"]
        list_sent_nums = thing["sentence_no"]
        list_sentences = thing["sentences"]
        for index in range(len(list_sent_nums)):
            embedded = calc_sent_embd(list_sentences[index])
            the_new_tuple = (embedded, list_sent_nums[index])
            tpls.append(the_new_tuple)
        tuple_id_embds = (id, tpls)
        list_with_isntances.append(tuple_id_embds)
    fp = open(path_out,"wb")
    pickle.dump(list_with_isntances,fp)
    fp.close()
    return list_with_isntances

class OneClusterModel():
    def __init__(self, comparer=None):
        if comparer is not None:
            self.comparer = comparer
        else:
            self.comparer = Comparer()
            print("No comparer found. Make a new one")


    def save(self, path):
        self.comparer.save(path)

    @classmethod
    def load(cls, path):
        model = Comparer.load(path)
        return cls(model)


    def fit(self,data, epochs = 20):
        self.comparer.fit(data, epochs = epochs)

    def predict_clusters(self,data):
        """
        Assume that data has the form of [(embd, sent_no)(embd, sent_no)(embd, sent_no)(embd, sent_no)(embd, sent_no)].
        This function predicts the clusters of an unlabeled instance.
        :param data: List containing tuples of embeddings and sentence numbers.
        :return: List containing the clusters as lists.
        """

        G = nx.Graph()
        for index, tuple in enumerate(data):
            embd = tuple[0]
            sent_no = tuple[1]
            for index_rest, other_tuple in enumerate(data[index:]):
                embd_other = other_tuple[0]
                sent_no_other = other_tuple[1]
                G.add_node(sent_no)
                G.add_node(sent_no_other)
                related = self.comparer.predict_same_cluster(embd,embd_other)
                if related:
                    G.add_edge(sent_no,sent_no_other)
        graphs = list(nx.connected_components(G))
        result = []
        for dic in graphs:
            result.append(sorted(list(dic)))
        return result

    def predict(self, data):
        """The data has something of the form of tuples
        (id, [(embd, sent_no)(.,..)(.,..)(.,..)(.,..)]) organized as a list. """
        preds = []
        for tuple in data:
            id = tuple[0]
            list_with_sentences = tuple[1]
            clusters = self.predict_clusters(list_with_sentences)
            new_dict = {"id":id, "pred_clusters":clusters}
            preds.append(new_dict)
        return preds

def read_pickle(path):
    """
    Reads in a pickle file with word embeddings. Should be an Array with embeddings.
    Returns an list with the embeddings as torch tensor.
    """
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))
    return data


def convert_to_scorch_format(docs, cluster_key="event_clusters"):
    # Merge all documents' clusters in a single list

    all_clusters = []
    for idx, doc in enumerate(docs):
        for cluster in doc[cluster_key]:
            all_clusters.append([str(idx) + "_" + str(sent_id) for sent_id in cluster])

    all_events = [event for cluster in all_clusters for event in cluster]
    all_links = sum([list(itertools.combinations(cluster,2)) for cluster in all_clusters],[])

    return all_links, all_events


def evaluate(goldfile, sysfile):
    """
    Uses scorch -a python implementaion of CoNLL-2012 average score- for evaluation. > https://github.com/LoicGrobol/scorch | pip install scorch
    Takes gold file path (.json), predicted file path (.json) and prints out the results.
	This function is the exact way the subtask3's submissions will be evaluated.
    """
    gold = read(goldfile)
    sys = read(sysfile)

    gold_links, gold_events = convert_to_scorch_format(gold)
    sys_links, sys_events = convert_to_scorch_format(sys, cluster_key="pred_clusters")

    with open("gold.json", "w") as f:
        json.dump({"type":"graph", "mentions":gold_events, "links":gold_links}, f)
    with open("sys.json", "w") as f:
        json.dump({"type":"graph", "mentions":sys_events, "links":sys_links}, f)

    subprocess.run(["scorch", "gold.json", "sys.json", "results.txt"])
    print(open("results.txt", "r").read())
    subprocess.run(["rm", "gold.json", "sys.json", "results.txt"])

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', '--train_file', required=True, help="The path to the training data json file")
    parser.add_argument('-prediction_output_file', '--prediction_output_file', required=True, help="The path to the prediction output json file")
    parser.add_argument('-test_file', '--test_file', required=False, help="The path to the test data json file")
    args = parser.parse_args()
    return args





def create_submission():
    """
    This code was used to produce the results of the paper.
    """
    data_es = read("Test_Data_Raw/test-es.json")
    data_en = read("Test_Data_Raw/test-en.json")
    data_pr = read("Test_Data_Raw/test-pr.json")

    t_en = generate_test_data(data_en, "Test_Data_Raw/ebbd-en.pkl")
    t_es = generate_test_data(data_es, "Test_Data_Raw/ebbd-es.pkl")
    t_pr = generate_test_data(data_pr, "Test_Data_Raw/ebbd-pr.pkl")

    # test_data = read_pickle("Test_Data_Raw/Embedded_Test_Data.pkl")

    train_ins = read_pickle("embd_en-train.pkl")
    print(len(train_ins[0]))
    one_isnt = train_ins[0]
    print(one_isnt)
    exit()

    model = OneClusterModel.load("comparer_model")
    model.comparer.model.summary()

    train_predictions = model.predict(t_en)
    with open("submission_en.json", "w", encoding="utf-8") as f:
        for doc in train_predictions:
            f.write(json.dumps(doc) + "\n")

    train_predictions = model.predict(t_es)
    with open("submission_es.json", "w", encoding="utf-8") as f:
        for doc in train_predictions:
            f.write(json.dumps(doc) + "\n")

    train_predictions = model.predict(t_pr)
    with open("submission_pr.json", "w", encoding="utf-8") as f:
        for doc in train_predictions:
            f.write(json.dumps(doc) + "\n")






def main(train_file,prediction_output_file,test_file=None):

    #Create model.
    model = OneClusterModel()

    #Read training data.
    train_data = read(train_file)

    #Transform the training data such we can train a net
    transformed_train_data = transfrom_to_sentence_embds(train_data, "train_data_embedded.pkl")
    transformed_train_data_predict = generate_test_data(train_data, "train_data_embedded_predict.pkl")

    #Fit.
    model.fit(transformed_train_data, epochs=20)
    model.save("comparer_model")


    # Predict train data so that we can evaluate our system
    train_predictions = model.predict(transformed_train_data_predict)
    with open(prediction_output_file, "w", encoding="utf-8") as f:
        for doc in train_predictions:
            f.write(json.dumps(doc) + "\n")

    # Evaluate sys outputs and print results.
    evaluate(train_file, prediction_output_file)

    # If there is a test file provided, make predictions and write them to file.
    if test_file:
        # Read test data.
        test_data = read(test_file)
        transformed_test_data = generate_test_data(test_data,"test_data_embedded_predict.pkl")
        # Predict and save your predictions in the required format.
        test_predictions = model.predict(transformed_test_data)
        with open("sample_submission.json", "w", encoding="utf-8") as f:
            for doc in test_predictions:
                f.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
    args = parse()
    main(args.train_file, args.prediction_output_file, args.test_file)





