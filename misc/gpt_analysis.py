# Standard imports
import os, sys
import pandas as pd
import openai
from tqdm import tqdm
import re

# Local imports
from word_norms import Word2Norm, clean_word
import myutils

WP_SOURCE = "simlex"
#WP_SOURCE = "all"
GPT_VERSION = "gpt3.5"
DF_FNAME = f"gpt_analysis/{GPT_VERSION}_{WP_SOURCE}.tsv"
ASK4REASON = True
if GPT_VERSION in ["gpt3.5"]:
    with open("/home/jumperkables/.openai/OPEN_API_KEY", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read()[:-1]
    openai.api_key = os.getenv("OPENAI_API_KEY")
FREQ = 20


def openai_response(prompt: str):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        #stop=["\n"]
    )
    resp = response["choices"][0]["text"]
    resp = re.split("\W+", resp)
    resp = [r for r in resp if r != ""]
    gptscore = [(resp[i+1], resp[i+2]) for i in range(0, 3*FREQ, 3)]
    reason = ["" for i in range(len(resp))]
    return gptscore, reason


GPT_SWITCH = {
    "gpt3.5": openai_response,
}


if WP_SOURCE == "all":
    norm_dict_path = os.path.join( os.path.dirname(__file__) , "all_norms.pickle")
    norm_dict = myutils.load_pickle(norm_dict_path)
    wps = norm_dict.word_pairs.keys()
    wps = [k.split("|") for k in wps]
elif WP_SOURCE == "simlex":
    simlex = pd.read_csv("/home/jumperkables/kable_management/data/a_vs_c/SimLex-999/SimLex-999.txt", sep='\t')
    wps = list(zip(simlex.word1, simlex.word2))
else:
    raise ValueError("Word pair source needed")

data = []


for i1 in tqdm(range(0, len(wps), FREQ), total=len(wps)//FREQ):
    sub_wps = wps[i1:i1+FREQ]
    #prompt = f"This question is about scoring the difference in how categorically related and how associated pairs of words are i.e. the difference between a SimLex999 score and USF association score respectively.\n"
    prompt = ""
    prompt += f"Q: Dual coding theory hypothesises that the human brain stores concrete concepts in categorically related structures, and abstract concepts in associative frameworks. This question requires an understanding of the difference between categorical relations, and associations between pairs of words. For each of the following pairs of words, please score on a likert scale of 1 (lowest) to 10 (highest) how categorically similar. Then, score on a likert scale of 1 (lowest) to 10 (highest) how associated they are. Carefully understand why these scores may be different. Please omit explanations."
    for i2, wp in enumerate(sub_wps):
        w1, w2 = wp
        prompt += f"\n{i2}: '{w1}' and '{w2}'"
    prompt += f"\nA:"
    gptscore, reason = GPT_SWITCH[GPT_VERSION](prompt)
    for i2, wp in enumerate(sub_wps):
        w1, w2 = wp
        data.append([w1, w2, gptscore[i2][0], gptscore[i2][1], reason[i2]])

df = pd.DataFrame(data, columns=["word1", "word2", "gptCtgrl", "gptAssoc", "reason"])
df.to_csv(DF_FNAME)
