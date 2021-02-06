# Concreteness infromation etc for the VQACP dataset
Concreteness study for VQACP/VQACP2

## Lxmert + Classifier:
The first stable model i got running on VQACP

## VQACP

### Unfreeze Heads
Training the attention heads as well as the classification layers
**Acc: 20%**

#### Language Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Language_vqacphighconc-mseqs.png)|![](lxmert+classifier/Language_vqacplowconc-m.png)

#### Vision Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Vision_vqacphighconc-mseqs.png)|![](lxmert+classifier/Vision_vqacplowconc-m.png)

#### Cross Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Cross_vqacphighconc-mseqs.png)|![](lxmert+classifier/Cross_vqacplowconc-m.png)



## VQACP2

### Unfreeze Heads
Training the attention heads as well as the classification layers
**Acc: 20%**

#### Language Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Language_vqacp2highconc-mseqs.png)|![](lxmert+classifier/Language_vqacp2lowconc-m.png)

#### Vision Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Vision_vqacp2highconc-mseqs.png)|![](lxmert+classifier/Vision_vqacp2lowconc-m.png)

#### Cross Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Cross_vqacp2highconc-mseqs.png)|![](lxmert+classifier/Cross_vqacp2lowconc-m.png)



### Unfreeze None
**Acc: 20%**
Only train the classification layers

