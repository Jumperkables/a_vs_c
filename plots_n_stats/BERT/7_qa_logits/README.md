# Transformer QA-variant Logit Information
The huggingface transformer pytorch library is a blessing. We consider a study similar to the "softmax response" experiments previously done. We calculate the number of classes required to hit each softmax threshold of the output layer of the QA models.

# LXMERT-QA:
## Softmax Threshold > 0.9
Conc Questions|Abs Questions
-|-
![pending](lxmertconcgt0pt95_softmax0.9.png)|![pending](lxmertconclt0pt3_softmax0.9.png)

## Softmax Threshold > 0.95
Conc Questions|Abs Questions
-|-
![pending](lxmertconcgt0pt95_softmax0.95.png)|![pending](lxmertconclt0pt3_softmax0.95.png)

## Softmax Threshold > 0.99
Conc Questions|Abs Questions
-|-
![pending](lxmertconcgt0pt95_softmax0.99.png)|![pending](lxmertconclt0pt3_softmax0.99.png)
