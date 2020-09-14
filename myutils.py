__author__ = "Jumperkables  "

import os, pickle, json


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
