## Instructions
Variations of (`plots_n_stats/tvqa/1_data_concreteness_distribution/get_tvqa_avc_stats.sh`) built on (`plots_n_stats/tvqa/1_data_concreteness_distribution/tvqa_avc_stats.py`)

# 'Concreteness Density' of Various Aspects of TVQA
Trying to find out how concrete or abstract words actually are in TVQA (MRC norms only at this point), we plotted the concreteness values of different parts of TVQA i.e. questions, subtitles, visual concepts, correct answers etc... We average the concreteness scores of every word in said categories, which I call `concreteness density`.

## Including Zero scores
Though many words in the TVQA question didn't have MRC norms (None), it turns out many of them had a score of 0, which on inspection I interpereted for some words means 'undecided'. These are the initial concreteness density calculations including the zeroes. What you are seeing below is the amount of, for example, questions that have a concreteness density in each of the specified ranges.

### Questions
Concreteness Range | % of Questions | | Question-type | Average Concreteness
:-- | :-- | --- | --: | --:
None    | 0.00828%  ||  What    |   186.40642986443427
0       | 11.10858% ||  Who     |   186.4392364619237
0-100   | 11.52193% ||  Why     |
100-200 | 33.87572% ||  Where   |
200-300 | 28.22551% ||  How     |
300-400 | 11.93044% ||  Which   |
400-500 | 2.78922%  ||  Other   |
500-600 | 0.48304%  ||          |
600-700 | 0.05727%  ||          |

Question-type | Average Concreteness
--- | ---
What | 186.40642986443427
Who | 186.4392364619237
Why | 181.52610320647253
Where | 170.28579106557763 
How | 177.98999847982503
Which | 227.06613767852085
Other | 178.44945282594983

## No Zero scores
As mentioned above, many words appear to have just been assigned a norm of 0. To offset any lazy-labelling weighing down our values, I ran the rest of these experiments ignoring the zero scores.

Questions (NO ZERO)
0 : 0.00000%
None : 11.11686%
0-100 : 0.00000%
100-200 : 0.00552%
200-300 : 3.76221%
300-400 : 18.76548%
400-500 : 32.54045%
500-600 : 25.21892%
600-700 : 8.59055%

Visual Concepts (NO ZERO)
0 : 0.00000%
None : 0.00483%
0-100 : 0.00000%
100-200 : 0.00000%
200-300 : 0.00000%
300-400 : 0.00000%
400-500 : 0.00207%
500-600 : 86.23538%
600-700 : 13.75772%

CORRECT ANSWERS (NO ZERO)
0 : 0.00000%
None : 30.21079%
0-100 : 0.00000%
100-200 : 0.01311%
200-300 : 3.82761%
300-400 : 12.65114%
400-500 : 16.90339%
500-600 : 24.27163%
600-700 : 12.12234%

Incorrect answers (NO ZERO)
0 : 0.00000%
None : 31.74675%
0-100 : 0.00000%
100-200 : 0.00856%
200-300 : 3.69887%
300-400 : 12.60288%
400-500 : 16.19541%
500-600 : 23.05906%
600-700 : 12.68847%

Spacy subject-object (NO ZERO)
0 : 0.00000%
None : 53.02567%
0-100 : 0.00000%
100-200 : 0.00000%
200-300 : 1.01535%
300-400 : 5.11756%
400-500 : 8.84902%
500-600 : 16.33234%
600-700 : 15.66005%

