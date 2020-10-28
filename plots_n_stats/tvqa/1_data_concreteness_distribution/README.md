## Instructions
Variations of (`plots_n_stats/tvqa/1_data_concreteness_distribution/get_tvqa_avc_stats.sh`) built on (`plots_n_stats/tvqa/1_data_concreteness_distribution/tvqa_avc_stats.py`)

# 'Concreteness Density' of Various Aspects of TVQA
Trying to find out how concrete or abstract words actually are in TVQA (MRC norms only at this point), we plotted the concreteness values of different parts of TVQA i.e. questions, subtitles, visual concepts, correct answers etc... We average the concreteness scores of every word in said categories, which I call `concreteness density`.

## Including Zero scores
Though many words in the TVQA question didn't have MRC norms (None), it turns out many of them had a score of 0, which on inspection I interpereted for some words means 'undecided'. These are the initial concreteness density calculations including the zeroes. What you are seeing below is the amount of, for example, questions that have a concreteness density in each of the specified ranges.

### Questions
Concreteness Range | % of Questions | | Question-type | Average Concreteness
:-- | :-- | --- | --: | --:
None    | 0.01%  ||  What    |   186.41
0       | 11.11% ||  Who     |   186.44
0-100   | 11.52% ||  Why     |   181.53
100-200 | 33.88% ||  Where   |   170.29
200-300 | 28.23% ||  How     |   177.99
300-400 | 11.93% ||  Which   |   227.07
400-500 | 2.79%  ||  Other   |   178.45
500-600 | 0.48%  ||          |
600-700 | 0.06%  ||          |


## No Zero scores
As mentioned above, many words appear to have just been assigned a norm of 0. To offset any lazy-labelling weighing down our values, I ran the rest of these experiments ignoring the zero scores.

Conc Range  | Questions | Vcpts     | Correct Answers   | Wrong Answers | Subj/Obj\*
:--         | --:       | --:       | --:               | --:           | --:
0           | 0%        | 0%        | 0%                | 0%            | 0%
None        | 11.12%    | 0.004%    | 30.21%            | 31.75%        | 53.03%
0-100       | 0%        | 0%        | 0%                | 0%            | 0%
100-200     | 0.01%     | 0%        | 0.01%             | 0.01%         | 0%
200-300     | 3.76%     | 0%        | 3.83%             | 3.70%         | 1.02%
300-400     | 18.77%    | 0%        | 12.65%            | 12.60%        | 5.12%
400-500     | 32.54%    | 0.002%    | 16.90%            | 16.20%        | 8.85%
500-600     | 25.22%    | 86.24%    | 24.27%            | 23.06%        | 16.33%
600-700     | 8.59%     | 13.76%    | 12.12%            | 12.69%        | 15.66%
\* = From the spacy tokeniser.
