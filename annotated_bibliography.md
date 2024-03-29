## Annotated Bibliography

**LANL-Chosen Papers:**

1. S. Gelman, S. A. Fahlberg, P. Heinzelman, P. A. Romero, A. Gitter, Neural networks to learn protein sequence–function relationships from deep mutational scanning data. Proceedings of the National Academy of Sciences. 118, e2104878118 (2021).
	
    In this paper, multiple neural network architectures are tested to investigate how a network’s internal representation affects its ability to learn protein sequence-function mapping. The authors presented a supervised learning framework capable of inferring this type of mapping from deep mutational scanning data. The background information on neural network architecture will be helpful for understanding how machine learning can be applied to understanding antibody and antigen interactions. This paper may have less relative importance, as it has only 8 citations and 66 references, however, it was published very recently. 

2. J. Graves, J. Byerly, E. Priego, N. Makkapati, S. V. Parish, B. Medellin, M. Berrondo, A Review of Deep Learning Methods for Antibodies. Antibodies (Basel). 9, 12 (2020).
    
    This review provides background on several deep learning algorithms that aim to predict protein and antibody interactions. It emphasizes that antigen and antibody interactions should be considered as a unique case of protein interactions. The discussions of training data of different deep learning methods will prove useful when curating our dataset. This paper has fairly high relative importance with 50 citations and 71 references, especially for only being published for two years.

3. B. Hie, E. D. Zhong, B. Berger, B. Bryson, Learning the language of viral evolution and escape. Science. 371, 284–288 (2021).
    
    This paper discusses how machine learning algorithms can predict viral escape patterns using sequence data and language models from three proteins: influenza hemagglutinin, HIV-1 envelope glycoprotein, and severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). This paper will be useful because it provides good background information and application practices for machine learning in antibody/antigen research. Considering the paper was published in the last year, it has high relative importance with 51 citations and 49 references.

**Software Papers:**

4. W. L. Hamilton, R. Ying, J. Leskovec, Inductive Representation Learning on Large Graphs (2018), (available at http://arxiv.org/abs/1706.02216).
	
    GraphSAGE is one of the machine learning architectures that will be explored in our project. This conference paper is by the GraphSAGE authors and provides background information on the framework, as well as extensive details on the algorithm they created. This will be a useful resource to reference when curating interaction data for this type of machine learning architecture. This paper was chosen as it references one of the machine learning architectures/softwares our mentors suggested we research, and its high number of citations at 1408 with 41 references.

5. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, PyTorch: An Imperative Style, High-Performance Deep Learning Library, 12.
    
    This paper discusses PyTorch, an architecture of the machine learning library, and its utilization as a deep learning framework. This paper introduces a useful resource that is both easy to use and extremely fast. Specifically, PyTorch is able to do complex computations while matching the performance of the fastest deep learning libraries. This paper was chosen as it is highly cited at 4,657 total citations and 43 cited references, and also contains information about PyTorch, which our mentors told us to research.

**Additional Papers:**

6. D. M. Fowler, S. Fields, Deep mutational scanning: a new style of protein science. Nat Methods. 11, 801–807 (2014).
	
    This method's paper provided an overview of deep mutational scanning, which utilizes high-throughput DNA sequencing to assess functional capacity of many protein variants simultaneously. Data generated from this method have been used to successfully train neural networks to learn sequence-function relationships in proteins. The methods described in this paper can be useful for understanding how sequence data can be used to inform machine learning for protein-protein interactions. This paper was selected as it has a high number of citations (456 citations, 52 references), indicating its importance to the field.

7. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, "Attention is All you Need" in Advances in Neural Information Processing Systems (Curran Associates, Inc., 2017; https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), vol. 30.
    
    In this conference paper, the Transformer neural network architecture model is introduced. Transformers rely on an attention mechanism rather than recurrence. The background information provided in this paper is particularly useful due to the increasing usage of transformers in natural language processing, which can be applied to machine learning algorithms for language models that consist of protein sequences (and so can be applied to antibody and antigen interactions). This paper was included as it was highly cited (9,897 citations, 32 references) and has background information on one of the neural network models that our mentors suggested we research. 

8. M. I. J. Raybould, A. Kovaltsuk, C. Marks, C. M. Deane, CoV-AbDab: the coronavirus antibody database. Bioinformatics. 37, 734–735 (2021).
    
    This paper provides background information on the coronavirus antibody database, or CoV-AbDab, which consolidates information on coronaviruses such as SARS-CoV2, SARS-CoV1, and MERS-CoV. Information present in this paper is useful for understanding the data within an antibody database and how to query it, which is particularly important for us as we are curating an antigen-antibody dataset using data from public databases. Additionally, this paper was found because we are utilizing this database to pull data from to build our own database for machine learning, and this is written by the authors of the database. It has 92 citations, and 0 references, which suggests it has relatively high importance given the publish date of last year.

9. R. Gao, T. Yang, Y. Shen, Y. Rong, K. Ye, J. Nie, RPI-MCNNBLSTM: BLSTM Networks Combining With Multiple Convolutional Neural Network Models to Predict RNA-Protein Interactions Using Multiple Biometric Features Codes. IEEE Access. 8, 189869–189877 (2020).
	
    This paper covers cross-validation, representing structure as matrices, and convolutional neural networks along with BLSTM. BLSTM is a recurrent neural network that our partners asked us to look into, and this paper provides some background. It will also be useful if our partners would like us to cross-validate our training data when curating the dataset. This paper was published recently, so it has a lower amount of citations (2 citations, 36 references). However, it covers a very specific topic, and relative to other papers on BLSTM networks, it has a fairly high amount of citations.

10. A. W. Senior, R. Evans, J. Jumper, J. Kirkpatrick, L. Sifre, T. Green, C. Qin, A. Žídek, A. W. R. Nelson, A. Bridgland, H. Penedones, S. Petersen, K. Simonyan, S. Crossan, P. Kohli, D. T. Jones, D. Silver, K. Kavukcuoglu, D. Hassabis, Improved protein structure prediction using potentials from deep learning. Nature. 577, 706–710 (2020).
	
    We found this article by looking at the most cited references from a mentor-provided paper, Graves et al., 2020. There are 974 citations for this paper and it includes 55 references. This article was most relevant to understanding how deep-learning can use sequence information to make predictions about the structure of a protein. AlphaFold uses a convolutional neural network trained on PDB structures to accurately predict the distances between pairs of amino acid residues in homologous structures.

11. A. J. Greaney, T. N. Starr, P. Gilchuk, S. J. Zost, E. Binshtein, A. N. Loes, S. K. Hilton, J. Huddleston, R. Eguia, K. H. D. Crawford, A. S. Dingens, R. S. Nargi, R. E. Sutton, N. Suryadevara, P. W. Rothlauf, Z. Liu, S. P. J. Whelan, R. H. Carnahan, J. E. J. Crowe, J. D. Bloom, Complete mapping of mutations to the SARS-CoV-2 spike receptor-binding domain that escape antibody recognition. bioRxiv (2020), doi:10.1101/2020.09.10.292078.
    
    We found this article by footnote chasing a given mentor article (Hie, et.,al, 2021) and it currently has been cited 145 times. This paper takes into account how antibodies that attack the same location on the spike protein will undergo different types of mutations. They designed a way to map amino acid mutations and described how that affects antibody binding. This is helpful for our project because we have to figure out a way to map our predicted mutations back to the specific antibodies with high accuracy. We can't use all of their methods because they make use of lab procedures such as electron microscopy which we do not have access to. However, they used structural epitopes to confirm the identity of escaped antigens and I think we can do something similar with AlphaFold. 

12. B. A. Sokhansanj, G. L. Rosen, Mapping Data to Deep Understanding: Making the Most of the Deluge of SARS-CoV-2 Genome Sequences. mSystems. 7, e00035-22 (2022).
	
    We found this article by looking among the articles most pertinent to our project that used the (Gelman et al., 2021) and (Graves et al., 2020) articles as references. This paper was published this year so it only has 1 citation so far, and includes 95 references. This one was the most relevant to our project goal of understanding how different machine learning architectures can be used to understand protein-protein interactions. This paper includes examples and references to additional studies where convolutional neural networks and Transformer-based architectures have been used to make predictions from sequence data. These are two of the machine learning architectures we are exploring in our project.

13. M. C. Maher, I. Bartha, S. Weaver, J. di Iulio, E. Ferri, L. Soriaga, F. A. Lempp, B. L. Hie, B. Bryson, B. Berger, D. L. Robertson, G. Snell, D. Corti, H. W. Virgin, S. L. Kosakovsky Pond, A. Telenti, Predicting the mutational drivers of future SARS-CoV-2 variants of concern. Science Translational Medicine. 14, eabk3445 (2022).
    
    We found this article by looking among the most cited articles that used the (Hie et al., 2021) article as a reference. This paper was published this year and already has 11 citations; it was listed among the highly cited papers.This study evaluated the predictive value of various biological features and were able to identify the primary biological drivers of Sars-Cov-2 short-term evolution. This is relevant to which dataset features will be most useful in training and testing our machine learning model to make predictions about viral evolution. 

