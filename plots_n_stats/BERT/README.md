# BERT Analysis
Analysis of BERT models inspired by [Hopfield Networks is All You Need](https://arxiv.org/pdf/2008.02217.pdf). The aim is to find the differences that may reside in BERT attention heads between concrete and abstract tokens.

## MT40k Top 3000 & Bottom 3000 Concretness:
Consider the number of logits in softmaxes of attention heads needed to reach 90% (intuitively, a measure of how focused on few or many tokens a given attention head is). A high number of logits required to sum to 0.9 means that there is some "metastable state" of the value in the key,value,query transformer triplet. Perhaps given the right conditions of abstractness and concreteness, we may see differences in how high or low this count is, potentially reflecting the associative or categorical nature implied in dual-coding theory.
(`BERT_mt40k_topbotk.sh`), you can vary k in the script easily.

**NOTE:** We process these as sequences of size 20 at a time, naturally extremely limited in an already rather blunt experiment. These are initial experimental results, more dedicated experiments with higher sequence lengths will take exponentially longer to process.

MT40k: Responses from heads of top 3000 concreteness tokens, split into sequences of 20 at a time:
![](bert_responses_top3000.png)

MT40k: Responses from heads of bottom 3000 concreteness tokens (most abstract):
![](bert_responses_top3000.png)
