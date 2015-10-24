# Deep-Semi-NMF
Theano-based implementation of Deep Semi-NMF.

George Trigeorgis, Konstantinos Bousmalis, Stefanos Zafeiriou, Bjoern W. Schuller 

Proceedings of The 31st International Conference on Machine Learning, pp. 1692–1700, 2014
http://jmlr.org/proceedings/papers/v32/trigeorgis14.html

A deep matrix factorization method for learning attribute representations

George Trigeorgis, Konstantinos Bousmalis, Stefanos Zafeiriou, Bjoern W.Schuller
http://arxiv.org/abs/1509.03248

Semi-Non-negative Matrix Factorization is a technique that learns a low-dimensional representation of a dataset that lends itself to a clustering interpretation. It is possible that the mapping between this new representation and our original data matrix contains rather complex hierarchical information with implicit lower-level hidden attributes, that classical one level clustering methodologies can not interpret. In this work we propose a novel model, Deep Semi-NMF, that is able to learn such hidden representations that allow themselves to an interpretation of clustering  according to different, unknown attributes of a given dataset. We also present a semi-supervised version of the algorithm, named Deep WSF, that allows the use of (partial) prior information for each of the known attributes of a dataset, that allows the model to be used on datasets with mixed attribute knowledge. Finally, we show that our models are able to learn low-dimensional representations that are better suited for clustering, but also classification, outperforming Semi-Non-negative Matrix Factorization, but also other state-of-the-art methodologies variants. 

Demo
====
For a quick practical example of Deep Semi-NMF please see [here] (Deep%20Semi-NMF.ipynb).
