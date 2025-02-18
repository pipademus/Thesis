# === IMPORTS === #
import numpy as np
import os
import pandas as pd
import psutil
import scipy.sparse as sp
import statistics
import tensorflow as tf
import threading
import time
from operator import itemgetter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import AGNNConv, EdgeConv, GATConv, GatedGraphConv, GeneralConv, GINConv, GraphSageConv, GTVConv, TAGConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# === INITIALISATION === #
cpu_resource = []
ram_resource = []
# change the name_X for the desired features
name_traceid = "traceId"
name_starttime = "startTime"
name_spanid = "id"
name_parentid = "pid"
name_services = ["serviceName", "dsName"] # the serviceName in the JDBC file is actually dsName for some reason
name_elapsedtime = "elapsedTime"
name_success = "success"
# change the neural network parameters as desired
learning_rate = 0.001
epochs = 400
es_patience = 10
batch_size = 32
folds = 10
# change the dataset folder and files as desired
data_folder = "/Datasets/AIOps_2020_05_31/"
current_folder = os.getcwd()
span_data_files = ["trace_csf.csv", "trace_fly_remote.csv", "trace_jdbc.csv", "trace_local.csv", "trace_osb.csv", "trace_remote_process.csv"]

def monitor_resources(interval = 5):
    """
    Function to check the resource consumption during the program's execution.

    Every 5 seconds, the CPU and RAM consumptions are collected and added to their respective list.
    """
    while(True):
        cpu_resource.append(psutil.cpu_percent(interval = interval))
        ram_resource.append(psutil.virtual_memory().percent)
        time.sleep(interval)

# === DATASET CREATION === #
def get_all_spans ():
    span_list = []
    for file in span_data_files:
        file_path = current_folder + data_folder + file
        try:
            file_data = pd.read_csv(file_path)
        except:
            print(f"Could not open '{file_path}'.")
            exit()
        for span in tqdm(range(len(file_data)), ascii = True, desc = f"Reading spans ('{file}')"):
            span_list.append(file_data.loc[span])
    sorted_span_list = sorted(span_list, key = itemgetter(name_traceid, name_starttime))
    return sorted_span_list

class AIOpsDataset (Dataset):
    """
    The AIOpsDataset class creates the dataset of graphs.
    
    Execution traces are passed to the AIOpsDataset instance, ordered by timestamp and trace identifier (if applicable). The traces within the same trace identifier are transformed into a graph, which is represented by the node-feature matrix, adjacency matrix, and node labels.
    """
    def __init__ (self, span_list, **kwargs):
        """Initialise the AIOpsDataset class, with Dataset as super class."""
        self.span_list = span_list
        super().__init__(**kwargs)

    def get_service (self, span):
        """
        Obtain the name of a service within the trace.

        Input:
            span: list. features of the span
        Output:
            callee: string. service name
        """
        if name_services[0] in span:
            callee = span[name_services[0]]
        else:
            callee = span[name_services[1]]
        return callee

    def create_graphs (self, span_list):
        """
        Create graphs from the list of spans.

        Input:
            span_list: list of lists. execution trace data
        Output:
            graphs: list of Graph. graphs representing execution trace data
        """
        traceid = 0
        graphs = []
        nodes = edges = elapsedTimes = services = success = []
        for span in tqdm(span_list, ascii = True, desc = "Graphing spans"):
            spanid = span[name_spanid]
            if traceid != span[name_traceid]:
                if nodes: # if the nodes list is not empty
                    graphs.append(self.format_graph(nodes, edges, elapsedTimes, services, success))
                nodes = []
                edges = []
                elapsedTimes = []
                services = []
                success = []
                traceid = span[name_traceid]
            else:
                parentid = span[name_parentid]
                edges.append((parentid, spanid))
            nodes.append(spanid)
            elapsedTimes.append(span[name_elapsedtime])
            services.append(self.get_service(span))
            success.append(span[name_success])
        return graphs

    def format_graph(self, nodes, edges, elapsedTimes, services, success):
        """
        Format the information within a trace identifier.

        Input:
            nodes: list of strings. the span identifiers of the spans within the trace
            edges: list of string tuples. the parent-child relationships of the trace
            elapsedTimes: list of floats. the execution times of each span within the trace
            services: list of strings. the service of each span within the trace
            success: list of booleans. the success status of each span within the trace
        Output:
            Graph(x, a, y): Graph. graph representation of the trace
        """
        node_number = len(nodes)
        feature_number = 3 # elapsedTime, serviceName, success
        nf_matrix = np.zeros((node_number, feature_number))
        nf_matrix[:, 0] = LabelEncoder().fit_transform(elapsedTimes)
        nf_matrix[:, 1] = LabelEncoder().fit_transform(services)
        nf_matrix[:, 2] = LabelEncoder().fit_transform(success)
        a_matrix = self.create_adjacency_matrix(nodes, edges)
        a_matrix = sp.csr_matrix(a_matrix)
        feature_count = nf_matrix.sum(0)
        node_labels = np.zeros((3,))
        node_labels[np.argmax(feature_count)] = 1
        return Graph(x = nf_matrix, a = a_matrix, y = node_labels)

    def create_adjacency_matrix (self, nodes, edges):
        """
        Create the adjacency matrix representing the connections between nodes.

        Input:
            nodes: list of strings. the span identifiers of the spans within the trace
            edges: list of string tuples. the parent-child relationships of the trace
        Output:
            adjacency: matrix. connected nodes have their connection represented as 1
        """
        node_length = len(nodes)
        adjacency = np.zeros((node_length, node_length), dtype = int)
        for i in range(node_length):
            for j in range(node_length):
                if (nodes[i], nodes[j]) in edges or (nodes[j], nodes[i]) in edges:
                    adjacency[i][j] = adjacency[j][i] = 1
        return adjacency
    
    def read (self):
        """
        The Dataset super class requires this function. Initialises construction of graphs.

        Output:
            graphs. list of Graph. graphs representing execution trace data
        """
        graphs = self.create_graphs(span_list)
        return graphs

span_list = get_all_spans()
data = AIOpsDataset(span_list = span_list, transforms=NormalizeAdj())

# === DATA MODEL === #
class Net(Model):
    """The Net class instantiates the neural network model."""
    def __init__(self):
        """Initialise the Net class, with Model as super class."""
        super().__init__()
        self.conv1 = AGNNConv(32, activation="relu") # relu, sigmoid, softplus, tanh
        #self.conv2 = AGNNConv(32, activation="relu")
        #self.conv3 = AGNNConv(32, activation="relu")
        #self.conv4 = EdgeConv(32, activation="relu")
        #self.conv5 = EdgeConv(32, activation="relu")
        #self.conv6 = EdgeConv(32, activation="relu")
        #self.conv7 = EdgeConv(32, activation="relu")
        #self.conv8 = EdgeConv(32, activation="relu")
        #self.conv9 = EdgeConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(data.n_labels, activation="softmax")

    def call(self, inputs):
        """
        The Model super class requires this function. Initialises the neural network layers.

        Input:
            inputs: tuple of tensors. the input features, adjacency matrix, and index tensor
        Output:
            output: symbolic tensor. predicted output probabilities
        """
        x, a, i = inputs
        x = self.conv1([x, a])
        #x = self.conv2([x, a])
        #x = self.conv3([x, a])
        #x = self.conv4([x, a])
        #x = self.conv5([x, a])
        #x = self.conv6([x, a])
        #x = self.conv7([x, a])
        #x = self.conv8([x, a])
        #x = self.conv9([x, a])
        output = self.global_pool([x, i])
        output = self.dense(output)
        return output

model = Net()
optimizer = Adam(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy()

idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]

loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

# === TRAINING AND EVALUATION === #
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    """
    Execute a training step in the model.

    Input:
        inputs: tuple of tensors. the input features, adjacency matrix, and index tensor
        target: tensor. the target labels for the input samples
    Output:
        loss: tensor. the computed loss value
        acc: tensor. the computed accuracy value
    """
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc

def evaluate(loader):
    """
    Evaluate the model on the dataset.

    Input:
        loader: DataLoader. the dataset object
    Output:
        avg_loss_acc[0]: tensor. the average loss computed over the dataset
        avg_loss_acc[1]: tensor. the average accuracy computed over the dataset
        class_report: dictionary. classification report containing precision, recall, and f1-score
    """
    output = []
    true_labels = []
    predicted_labels = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        true_labels.extend(tf.argmax(target, axis=1).numpy())
        predicted_labels.extend(tf.argmax(pred, axis=1).numpy())
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            avg_loss_acc = np.average(output[:, :-1], 0, weights=output[:, -1])
            c_matrix = confusion_matrix(true_labels, predicted_labels, normalize = "all")
            class_report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
            return avg_loss_acc[0], avg_loss_acc[1], class_report, c_matrix

epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
train_accuracy = []
train_loss = []
validation_accuracy = []
validation_loss = []
monitor_thread = threading.Thread(target = monitor_resources)
monitor_thread.daemon = True
monitor_thread.start()    # resource consumption
train_start = time.time() # time consumption
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    train_loss.append(loss)
    train_accuracy.append(acc)
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1
        
        val_loss, val_acc, val_class, val_c_matrix = evaluate(loader_va)
        validation_loss.append(val_loss)
        validation_accuracy.append(val_acc)
        """
        Uncomment this block for epoch-wise training and validation data
        print(
            "Epoch {} - Training Loss: {:.5f} - Training Accuracy: {:.5f}".format(
                epoch, np.mean(train_loss), np.mean(train_accuracy)
            )
        )
        print(
            "Validation Loss: {:.5f} - Validation Accuracy: {:.5f}".format(
                val_loss, val_acc
            )
        )
        """

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            #print("New best val_loss {:.5f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
        if patience == 0 and best_val_loss < 0.01:
            #print("Early stopping (best val_loss: {})".format(best_val_loss))
            break
train_end = time.time()
print(
    "Epochs {} - Training Loss: {:.5f} - Training Accuracy: {:.5f} - Validation Loss: {:.5f} - Validation Accuracy: {:.5f}".format(
        epoch, np.mean(train_loss), np.mean(train_accuracy), np.mean(validation_loss), np.mean(validation_accuracy),
    )
)
print(f"Train CPU Percentage: {statistics.fmean(cpu_resource)}%.")
print(f"Train RAM Percentage: {statistics.mean(ram_resource)}%.")
print(f"Train Elapsed Time: {train_end - train_start} seconds.")

# === RESULTS === #
cpu_resource = []
ram_resource = []
test_start = time.time() # time consumption
test_loss, test_acc, test_class, c_matrix = evaluate(loader_te)
print(
    "Testing Loss: {:.5f} - Testing Accuracy: {:.5f} - Testing Precision: {:.5f} - Testing Recall: {:.5f} - Testing F1-Score: {:.5f}".format(
        test_loss, test_acc, test_class["weighted avg"]["precision"], test_class["weighted avg"]["recall"], test_class["weighted avg"]["f1-score"]
    )
)
print(c_matrix)
test_end = time.time()
if (cpu_resource):
    print(f"Test CPU Percentage: {statistics.fmean(cpu_resource)}%.")
if (ram_resource):
    print(f"Test RAM Percentage: {statistics.mean(ram_resource)}%.")
print(f"Test Elapsed Time: {test_end - test_start} seconds.")