from itertools import combinations
from collections import Counter
import os, sys
from tqdm import tqdm
from statistics import mean
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import misc.myutils as myutils
from .word_norms import word_is_cOrI

######################################
######################################
######################################
######################################
################ Exerpts from the 'multimodal' pypip module
import re
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]
commaStrip = re.compile(r"(\d)(\,)(\d)")
periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")


def processPunctuation(inText):
    outText = inText
    if re.search(commaStrip, inText) != None:
        outText.replace(",", "")

    for p in punct:
        if p + " " in inText or " " + p in inText:
            outText = outText.replace(p, "")
        elif p in inText:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText



class EvalAIAnswerProcessor:
    """Processes an answer similar to Eval AI
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]


    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


def process_annotations(annotations_train, annotations_val, path_train, path_val, path_answers, args, norm_dict):
    """Process answers to create answer tokens,
    and precompute VQA score for faster evaluation.
    This follows the official VQA evaluation tool.
    """
    top_k_flag = (args.topk != -1)
    min_ans_occ_flag = not top_k_flag
    top_k = args.topk
    min_ans_occ = args.min_ans_occ
    if not(os.path.exists(path_train)) or not(os.path.exists(path_val)):
        print("Proccessed annotation file doesnt exist yet, create them...")
        all_annotations = annotations_train + annotations_val

        print("Processing annotations")
        processor = EvalAIAnswerProcessor()

        print("\tPre-Processing answer punctuation")
        for annot in tqdm(all_annotations):

            annot["multiple_choice_answer"] = processor(
                annot["multiple_choice_answer"]
            )
            # vqa_utils.processPunctuation(
            #     annot["multiple_choice_answer"]
            # )
            for ansDic in annot["answers"]:
                ansDic["answer"] = processor(ansDic["answer"])

        print("\tPre-Computing answer scores")
        for annot in tqdm(all_annotations):
            annot["scores"] = {}
            unique_answers = set([a["answer"] for a in annot["answers"]])
            for ans in unique_answers:
                scores = []
                # score is average of 9/10 answers
                for items in combinations(annot["answers"], 9):
                    matching_ans = [item for item in items if item["answer"] == ans]
                    score = min(1, float(len(matching_ans)) / 3)
                    scores.append(score)
                annot["scores"][ans] = mean(scores)
        print(f"Saving processed annotations at {path_train} and {path_val}")

        with open(path_train, "w") as f:
            json.dump(annotations_train, f)
        with open(path_val, "w") as f:
            json.dump(annotations_val, f)
    else:
        print("Annotations are preprocced, load them and create ans2idx...")
    #####################################
    # Processing min occurences of answer
    #####################################
    print(f"Removing uncommon answers")
    annotations_train = myutils.load_json(path_train)
    annotations_val = myutils.load_json(path_val)
    all_annotations = annotations_train + annotations_val

    occ = Counter(annot["multiple_choice_answer"] for annot in all_annotations)
    remove_ans = []
    if args.norm_ans_only:
        # Ignore all questions with answers that are not themselves a psycholinguistic conc/imag norm
        occ = {ans:value for ans,value in occ.items() if word_is_cOrI(norm_dict, ans)}

    answers = [ans for ans in occ if occ[ans] >= min_ans_occ]
    top_k_answers = occ.most_common(top_k)

    threshold_answers_path = f"{path_answers}/{'normAnsOnly_' if args.norm_ans_only else ''}occ_gt{min_ans_occ}_answers.json"
    topk_answers_path = f"{path_answers}/{'normAnsOnly_' if args.norm_ans_only else ''}top{top_k}_answers.json"
    print(f"Saving answers at {path_answers}")
    print(f"Top {top_k} answers: {topk_answers_path}. Threshold > {min_ans_occ} answers:{threshold_answers_path}")
    #assert ('yes' in top_k_answers) and ('no' in top_k_answers), f"yes and no need to be in the top 1000 answers"
    if min_ans_occ_flag:
        with open(threshold_answers_path, "w") as f:
            json.dump(answers, f)
    else:
        with open(topk_answers_path, "w") as f:
            json.dump(top_k_answers, f)
######################################
######################################
######################################
######################################
