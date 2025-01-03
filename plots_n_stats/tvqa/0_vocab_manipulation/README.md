### Instructions:
0. Edit the TVQA word2index vocabulary file in advance with (`tvqa/change_tvqa_w2idx.py`)
1. (`plots_n_stats/tvqa/0_vocab_manipulation/violin_plot.sh`) or (`plots_n_stats/tvqa/0_vocab_manipulation/qtype_plot.sh`)

# Question-type distribution:
The performance gain/loss on each question type (versus the models own average, seen in the brackets in the descriptions on the left)
![Browser isnt rendering](qtype/conc_hmap_full.png)

# TVQA feature responses:
(like the violins in our first paper)

## Disregarding Stopwords
The conditions on words that we KEEP. E.g: **concgt300** = ***KEEP*** words of concreteness ***GREATER-THAN 300***. (Note, scores are between 100-700, an upscaled version of the 1-7 likert scale used in MRC, I reckon this is because they don't like floating point numbers).
SVI | VI
--- | ---
![](violins/disregarding_stopwords/svi_concgt300.png) | ![](violins/disregarding_stopwords/vi_concgt300.png)
![](violins/disregarding_stopwords/svi_concgt500.png) | ![](violins/disregarding_stopwords/vi_concgt500.png)
![](violins/disregarding_stopwords/svi_conclt300.png) | ![](violins/disregarding_stopwords/vi_conclt300.png)
![](violins/disregarding_stopwords/svi_conclt500.png) | ![](violins/disregarding_stopwords/vi_conclt500.png)

## Clumsily including the concreteness of stopwords (Before) 
SVI | VI
--- | ---
![](violins/before_stopwords/tpfp_concgt300_svi_glove.png) | ![](violins/before_stopwords/tpfp_concgt300_svi_glove.png)
![](violins/before_stopwords/tpfp_concgt500_svi_glove.png) | ![](violins/before_stopwords/tpfp_concgt500_svi_glove.png)
![](violins/before_stopwords/tpfp_conclt300_svi_glove.png) | ![](violins/before_stopwords/tpfp_conclt300_svi_glove.png)
![](violins/before_stopwords/tpfp_conclt500_svi_glove.png) | ![](violins/before_stopwords/tpfp_conclt500_svi_glove.png)

![](violins/before_stopwords/tnfn_concgt300_svi_glove.png)
![](violins/before_stopwords/tnfn_concgt500_svi_glove.png)
![](violins/before_stopwords/tnfn_conclt300_svi_glove.png)
![](violins/before_stopwords/tnfn_conclt500_svi_glove.png)
