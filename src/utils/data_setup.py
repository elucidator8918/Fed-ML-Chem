"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import random_split, TensorDataset, Dataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from .common import *

NUM_WORKERS = os.cpu_count()

# Normalization values for the different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'animaux': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'breast': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'histo': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'MRI': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'DNA': None,
    'PCOS': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'MMF': None,
    'DNA+MRI' : dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'PILL' : dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'HIV': None
    }

class MultimodalDataset(Dataset):
    def __init__(self, dna_dataset, mri_dataset):
        self.dna_dataset = dna_dataset
        self.mri_dataset = mri_dataset
        self.length = min(len(self.dna_dataset), len(self.mri_dataset))
        self.dna_indices = list(range(len(self.dna_dataset)))
        self.mri_indices = list(range(len(self.mri_dataset)))
        if len(self.dna_dataset) > self.length:
            self.dna_indices = random.sample(self.dna_indices, self.length)
        if len(self.mri_dataset) > self.length:
            self.mri_indices = random.sample(self.mri_indices, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dna_data, dna_label = self.dna_dataset[self.dna_indices[idx]]
        mri_data, mri_label = self.mri_dataset[self.mri_indices[idx]]
        
        return (mri_data, dna_data), (mri_label, dna_label)
    
def read_and_prepare_data(file_path, seed, size=6, model_name='all-MiniLM-L6-v2'):
    """
    Reads DNA sequence data from a text file and prepares it for modeling.
    """
    # Read data from file
    data = pd.read_table(file_path).head(1000)

    # Function to extract k-mers from a sequence
    def getKmers(sequence):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

    # Function to preprocess data
    def preprocess_data(data):
        data['words'] = data['sequence'].apply(lambda x: getKmers(x))
        data = data.drop('sequence', axis=1)
        return data

    # Preprocess data
    data = preprocess_data(data)

    def kmer_lists_to_texts(kmer_lists):
        return [' '.join(map(str, l)) for l in kmer_lists]

    data['texts'] = kmer_lists_to_texts(data['words'])

    # Prepare data for modeling
    texts = data['texts'].tolist()
    y_data = data.iloc[:, 0].values
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(texts, convert_to_tensor=True).cpu().numpy()
    del embed_model
    del SentenceTransformer
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_data, test_size=0.2, random_state=seed)
        
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)
    return trainset, testset

def preprocess_graph():
    """
    Preprocess the HIV dataset from MoleculeNet for a classification task.

    The HIV dataset, introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, contains 
    data on over 40,000 compounds tested for their ability to inhibit HIV replication. The compounds 
    are categorized into three classes based on their activity: confirmed inactive (CI), confirmed active 
    (CA), and confirmed moderately active (CM). For this classification task, we combine the active 
    categories (CA and CM) and classify the compounds into two categories: inactive (CI) and active (CA and CM).

    The function splits the dataset into training and test sets, with 80% of the data used for training 
    and 20% for testing.

    Returns:
        tuple: A tuple containing the training set and the test set, each as a subset of the HIV dataset.
    """
    data = MoleculeNet(root="data", name="HIV")
    split_index = int(len(data) * 0.8)
    trainset, testset = data[:split_index], data[split_index:]

    return trainset, testset

def preprocess_and_split_data(au_mfcc_path):
    # Load the Audio+Vision(MP4 Video input divided into Audio and Images) data
    with open(au_mfcc_path, 'rb') as f:
        au_mfcc = pickle.load(f)

    # Initialize lists for data and labels
    data = []
    labels = []

    # Process the data
    for key in au_mfcc:
        emotion = int(key.split('-')[2]) - 1
        labels.append(emotion)
        data.append(au_mfcc[key])

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels).reshape(-1, 1)

    # Concatenate data and labels
    data = np.hstack((data, labels))

    # Shuffle data
    data = shuffle(data)

    # Split data and labels
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # One-hot encode labels
    num_classes = np.unique(y).size
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    # Split into test, train, and dev sets
    test_data = X[-181:-1]
    test_labels = y_one_hot[-181:-1]
    data = X[:-180]
    labels = y_one_hot[:-180]
    train_data = X[:1020]
    train_labels = y_one_hot[:1020]
    dev_data = X[1020:]
    dev_labels = y_one_hot[1020:]

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    dev_data_tensor = torch.tensor(dev_data, dtype=torch.float32)
    dev_labels_tensor = torch.tensor(dev_labels, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
    trainset = TensorDataset(train_data_tensor, train_labels_tensor)
    devset = TensorDataset(dev_data_tensor, dev_labels_tensor)
    testset = TensorDataset(test_data_tensor, test_labels_tensor)

    return trainset, devset, testset

def split_data_client(dataset, num_clients, seed):
    """
    This function is used to split the dataset into train and test for each client.
    :param dataset: the dataset to split (type: torch.utils.data.Dataset)
    :param num_clients: the number of clients
    :param seed: the seed for the random split
    """
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths += [len(dataset) - sum(lengths)]
    ds = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return ds

# Define model, architecture and dataset
# The DataLoaders downloads the training and test data that are then normalized.
def load_datasets(num_clients: int, batch_size: int, resize: int, seed: int, num_workers: int, splitter=10,
                  dataset="cifar", data_path="./data/", data_path_val=""):
    """
    This function is used to load the dataset and split it into train and test for each client.
    :param num_clients: the number of clients
    :param batch_size: the batch size
    :param resize: the size of the image after resizing (if None, no resizing)
    :param seed: the seed for the random split
    :param num_workers: the number of workers
    :param splitter: percentage of the training data to use for validation. Example: 10 means 10% of the training data
    :param dataset: the name of the dataset in the data folder
    :param data_path: the path of the data folder
    :param data_path_val: the absolute path of the validation data (if None, no validation data)
    :return: the train and test data loaders
    """
    DataLoader = PyGDataLoader if dataset == "hiv" else TorchDataLoader
    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])] if dataset not in ["MMF", "DNA", "hiv"] else None
    print(dataset)

    if dataset == "cifar":
        # Download and transform CIFAR-10 (train and test)
        transformer = transforms.Compose(
            list_transforms
        )
        trainset = datasets.CIFAR10(data_path + dataset, train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(data_path + dataset, train=False, download=True, transform=transformer)
    
    elif dataset == "hiv":
        trainset, testset = preprocess_graph()    

    elif dataset == "DNA":
        trainset, testset = read_and_prepare_data(data_path + dataset + '/human.txt', seed)        

    elif dataset == "MMF":        
        trainset, valset, testset = preprocess_and_split_data(data_path + dataset + '/Audio_Vision_RAVDESS.pkl')                    
    
    elif dataset == "DNA+MRI":        
        dataset_dna, dataset_mri = dataset.split("+")         
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms            

        transformer = transforms.Compose(list_transforms)
        supp_ds_store(data_path + dataset_mri)
        supp_ds_store(data_path + dataset_mri + "/Training")
        supp_ds_store(data_path + dataset_mri + "/Testing")
        trainset_mri = datasets.ImageFolder(data_path + dataset_mri + "/Training", transform=transformer)
        testset_mri = datasets.ImageFolder(data_path + dataset_mri + "/Testing", transform=transformer)
        trainset_dna, testset_dna = read_and_prepare_data(data_path + dataset_dna + '/human.txt', seed)
        trainset = MultimodalDataset(trainset_dna, trainset_mri)
        testset = MultimodalDataset(testset_dna , testset_mri)

    else:
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms

        transformer = transforms.Compose(list_transforms)
        supp_ds_store(data_path + dataset)
        supp_ds_store(data_path + dataset + "/Training")
        supp_ds_store(data_path + dataset + "/Testing")
        trainset = datasets.ImageFolder(data_path + dataset + "/Training", transform=transformer)
        testset = datasets.ImageFolder(data_path + dataset + "/Testing", transform=transformer)

    if dataset == "DNA":
        print("The training set is created for the classes: ('0', '1', '2', '3', '4', '5', '6')")
    elif dataset == "MMF":
        print("The training set is created for the classes: ('happy', 'sad', 'angry', 'fearful', 'surprise', 'disgust', 'calm', 'neutral')")        
    elif dataset == "DNA+MRI":
        print("The training set is created for the classes: ")
        print("('glioma', 'meningioma', 'notumor', 'pituitary')")
        print("('0', '1', '2', '3', '4', '5', '6')")
    elif dataset == "hiv":
        print("The training set is created for the classes: ('confirmed inactive (CI)', 'confirmed active (CA)/confirmed moderately active (CM)')")
    else:
        print(f"The training set is created for the classes : {trainset.classes}")        

    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets_train = split_data_client(trainset, num_clients, seed)
    if dataset == "MMF":
        datasets_val = split_data_client(valset, num_clients, seed)
    elif data_path_val and dataset not in ["DNA", "hiv"]:
        valset = datasets.ImageFolder(data_path_val, transform=transformer)
        datasets_val = split_data_client(valset, num_clients, seed)    
        
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        if dataset == "MMF" or data_path_val:
            # Use provided validation dataset
            trainloaders.append(DataLoader(datasets_train[i], batch_size=batch_size, shuffle=dataset != "MMF"))
            valloaders.append(DataLoader(datasets_val[i], batch_size=batch_size))
        else:            
            len_val = int(len(datasets_train[i]) * splitter / 100)  # splitter % validation set
            len_train = len(datasets_train[i]) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(datasets_train[i], lengths, torch.Generator().manual_seed(seed)) 
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader