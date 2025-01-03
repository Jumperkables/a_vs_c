__author__ = "Jumperkables  "
import torch
import torch.nn as nn
import os, pickle, json, statistics, math, re
import string
import numpy as np
import seaborn as sns
import torch
from nltk.corpus import stopwords
stpwrds = stopwords.words("english")

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: '{getattr(args, arg)}'")

def assert_torch(tensor):
    """
    If this is a numpy tensor, convert it to Torch and pass it back
    """
    if type(tensor) == np.ndarray:
        return torch.from_numpy(tensor)
    elif type(tensor) == torch.Tensor:
        return tensor
    else:
        raise ValueError(f"Unhandled data type: {type(tensor)}")

def remove_stopwords(sentence):
    return " ".join([word for word in sentence if word not in stpwrds])

def list_avg(lyst):
    if lyst == []:
        return 0
    else:
        return sum(lyst)/len(lyst)

def clean_word(word):
    if word != word:
        return None
    word = word.lower()
    word = word.translate(str.maketrans('', '', string.punctuation))
    #TODO deprecated??word = re.sub('[^a-z0-9]+', '', word)
    return word

def read_json_lines(file_path):
    with open(file_path, "r") as f:
        lines = []
        for l in f.readlines():
            loaded_l = json.loads(l.strip("\n"))
            lines.append(loaded_l)
    return lines

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def save_json_tab(data, file_path):
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_pickle(data, data_path):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

class MyCustomUnpickler(pickle.Unpickler):
    """
    Kindly borrowed from stack overflow: https://stackoverflow.com/questions/50465106/attributeerror-when-reading-a-pickle-file
    """
    def find_class(self, module, name):
        if module == "__main__":
            module = "misc.word_norms"
        return super().find_class(module, name)

class Identity(nn.Module):
    """
    A cool trick from an answer from ptrblck on pytorch forums https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
    Use this to replace layers in pytorch with simple identity layers. Much easier than trying to remove certain layers
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def load_norms_pickle(file_path):
    """
    Use for when you get an AttributeError:
        AttributeError: Can't get attribute 'CLASS' on <module '__main__'>
    """
    with open(file_path, "rb") as f:
        unpickler = MyCustomUnpickler(f)
        obj = unpickler.load()
        return obj

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def df_firstrow_to_header(df):
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header #set the header row as the df header
    return df

def n_softmax_threshold(array, threshold=0.9):
    """
    Calculate the minimum number of elements in a list that sum to the threshold value
    - List should be normalised, but will be if needed
    """
    #assert sum(array)==1, f"Array should be normalised s.t. all elements sum to 1"
    #import ipdb; ipdb.set_trace()
    array = np.array(array)
    array = array/np.sum(array)
    array = np.sort(array)[::-1]
    thresh = 0
    counting = 0
    for num in array:
        thresh += num
        counting += 1
        if thresh >= threshold:
            break
    return counting


class RUBi_Criterion(nn.Module):

    def __init__(self, loss_type="CrossEntropyLoss", mode="default"):
        """
        My implementation of the Reducing Unimodal Bias loss strategy introduced here: https://github.com/cdancette/rubi.bootstrap.pytorch
            loss_type   : Should the losses be CrossEntropyLoss, or BCEWithLogitsLoss etc..? 
            mode        : To be used to expand implementation if needed
        """
        super().__init__()
        assert mode in ["default"], f"Mode :`{mode}` not implemented"
        self.mode = mode
        assert loss_type in ["CrossEntropyLoss","BCEWithLogitsLoss"], f"loss_type: `{loss_type}` not implemented"
        if loss_type == "CrossEntropyLoss":
            self.main_loss = nn.CrossEntropyLoss(reduction='none')
            self.biased_loss = nn.CrossEntropyLoss(reduction='none')
            self.combined_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.main_loss = nn.BCEWithLogitsLoss(reduction='none')
            self.biased_loss = nn.BCEWithLogitsLoss(reduction='none')
            self.combined_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, main_logits, biased_logits, labels, biased_loss_weighting=1.0):
        """
            main_logits             : Pre-softmax class logits from the designated 'main' models
            biased_logits           : Pre-softmax class logits from the designated purposely biased model
            labels                  : Class labels (as if this were cross entropy loss)
            biased_loss_weighting   : The weight with which to amplify the main loss with biased
        """
        assert main_logits.shape == biased_logits.shape, f"Main and biased class logits are not the same shape. {main_logits.shape} and {biased_logits.shape}"
        out = {}
        # Combined mask
        biased_mask = torch.sigmoid(biased_logits)
        combined_logits = main_logits * (biased_mask*biased_loss_weighting)
        combined_loss_out = self.combined_loss(combined_logits, labels)
        #
        main_loss_out = self.main_loss(main_logits, labels)
        biased_loss_out = self.biased_loss(biased_logits, labels)
        out['combined_loss'] = combined_loss_out
        out['main_loss'] = main_loss_out
        out['biased_loss'] = biased_loss_out
        return out
        




def colour_violin(array, mode="median", max_x=550):
    assert mode in ["mean","median","mode"], f"{mode} not implemented. choose 'mean', 'median' or 'mode'"
    array = sorted(array, reverse=True)
    if mode == "mean":
        value = round(sum(array)/len(array))
    elif mode == "median":
        """
        Here the median of even numbered list simply picks the first of the 2 middle elements instead of averaging
        """
        value = array[len(array)//2]
    elif "mode":
        value = statistics.mode(array)
    else:
        raise ValueError(f"Impossible. Mode:{mode} should have been caught by the assertion")
    # Choose appropriate colour based on value's intensity

    colour = ["aquamarine","lightgreen","palegoldenrod","sandybrown","lightcoral"]
    assert value >= 0, f"Somehow the calculated value is below 0"
    colour = colour[math.floor(len(colour)*value/max_x)]
    ax = sns.violinplot(x=array, color='lightgray', linewidth=0.5, inner="quartile")#"lightgray")
    #sns.set(rc={'axes.facecolor':colour})#, 'figure.facecolor':'cornflowerblue'})
    ax.set(xlim=(-1, max_x))
    ax.set_facecolor(colour)
    ax.xaxis.set_label_position('top') 
    ax.annotate(f"{value}", xy=(7*max_x/10,-0.32), xytext=(7*max_x/10,-0.32), fontsize=8)
    return ax
