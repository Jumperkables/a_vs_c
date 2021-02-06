# Concreteness infromation etc for the GQA dataset
Concreteness study for GQA

## Lxmert + Classifier:
The first stable model i got running on GQA

### Unfreeze Heads
Training the attention heads as well as the classification layers
**Acc: 20%**

#### Language Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Language_gqahighconc-mseqs.png)|![](lxmert+classifier/Language_gqalowconc-m.png)

#### Vision Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Vision_gqahighconc-mseqs.png)|![](lxmert+classifier/Vision_gqalowconc-m.png)

#### Cross Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Cross_gqahighconc-mseqs.png)|![](lxmert+classifier/Cross_gqalowconc-m.png)


### Unfreeze None
**Acc: 20%**
Only train the classification layers

