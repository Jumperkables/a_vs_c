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
![](lxmert+classifier/Language_vqacp-topk500highconc-mseqs.png)|![](lxmert+classifier/Language_vqacp-topk500lowconc-m.png)

#### Vision Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Vision_vqacp-topk500highconc-mseqs.png)|![](lxmert+classifier/Vision_vqacp-topk500lowconc-m.png)

#### Cross Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Cross_vqacp-topk500highconc-mseqs.png)|![](lxmert+classifier/Cross_vqacp-topk500lowconc-m.png)



## vqacp-topk5002

### Unfreeze Heads
Training the attention heads as well as the classification layers
**Acc: 20%**

#### Language Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Language_vqacp2-topk500highconc-mseqs.png)|![](lxmert+classifier/Language_vqacp2-topk500lowconc-m.png)

#### Vision Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Vision_vqacp2-topk500highconc-mseqs.png)|![](lxmert+classifier/Vision_vqacp2-topk500lowconc-m.png)

#### Cross Attentions
Conc Pool|Abs Pool
-|-
![](lxmert+classifier/Cross_vqacp2-topk500highconc-mseqs.png)|![](lxmert+classifier/Cross_vqacp2-topk500lowconc-m.png)



### Unfreeze None
**Acc: 20%**
Only train the classification layers

