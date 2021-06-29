# Personalized Degrees

This repo includes the Python implementation as well as the code to replicate all experiments 
in the paper [Personalized Degrees: Effects on Link Formation in Dynamic Networks from an Egocentric Perspective](https://arxiv.org/pdf/1712.01796.pdf), 
presented at MSM workshop at the Web Conference (WWW) 2019.

## Abstract
Understanding mechanisms driving link formation in dynamic social networks is a long-standing problem that has implications to understanding social structure 
as well as link prediction and recommendation. Social networks exhibit a high degree of transitivity, which explains the successes of common neighbor-based 
methods for link prediction. In this paper, we examine mechanisms behind link formation from the perspective of an ego node. We introduce the notion of 
personalized degree for each neighbor node of the ego, which is the number of other neighbors a particular neighbor is connected to. 
From empirical analyses on four on-line social network datasets, we find that neighbors with higher personalized degree are more likely to 
lead to new link formations when they serve as common neighbors with other nodes, both in undirected and directed settings. This is complementary to the 
finding of Adamic and Adar [1] that neighbor nodes with higher (global) degree are less likely to lead to new link formations. Furthermore, on directed 
networks, we find that personalized out-degree has a stronger effect on link formation than personalized in-degree, whereas global in-degree has a stronger 
effect than global out-degree. We validate our empirical findings through several link recommendation experiments and observe that incorporating both 
personalized and global degree into link recommendation greatly improves accuracy.

## Contact
Please contact us if you have any questions or to report an issue. You can find the contact information of all 
authors in the [paper](https://arxiv.org/pdf/1712.01796.pdf).

> This repository has been published for the sole purpose of providing more information on the aforementioned publication.
