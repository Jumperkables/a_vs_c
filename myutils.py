__author__ = "Jumperkables  "

import os, pickle, json, statistics, math, re
import seaborn as sns

def clean_word(word):
    if word != word:
        return None
    word = word.lower()
    word = re.sub('[^a-z0-9]+', '', word)
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
    if sum(array) != 1:
        total = sum(array)
        array = [i/total for i in array]
    array = sorted(array, reverse=True)
    #import ipdb; ipdb.set_trace()
    counting = []
    for num in array:
        counting.append(num)
        if sum(counting) >= threshold:
            break
    return len(counting)

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
