# Machine Learning Approach to a Novel Curated Covid-19 Antibody Database
### By Toni Brooks, Matt Esqueda, Rachel Ferina, and Kaetlyn Gibson


## Overview
We cleaned four publicly available Covid-19 antibody-antigen datasets to curate a MongoDB database for machine learning.
We developed a binary classification machine learning model to identify if an antibody is SARS-CoV-2 or not, as well as a regression machine learning model to predict antigen binding affinity.


## Database Cleaning
**AlphaSeq:**
The AlphaSeq data was generated from in silico experiments, with Alphaseq assays used to collect the antibodies targeted against SARS-CoV-2. 
Alphaseq assays measure protein-protein interactions via the frequency of barcode pairs after next generation sequencing, with a higher frequency indicating stronger protein-protein interactions. The barcodes along with controls are used to estimate binding affinity, which is a numerical representation of how likely an antibody and antigen are to interact.


This dataset contains single chain fragment variable antibodies, or scFV, which are recombinant antibodies that contain one light chain and one heavy chain connected by a linker peptide. However this is a significant structural difference, and these single chain antibodies mean this data cannot be combined with the other datasets for machine learning.


The AlphaSeq dataset contains three replicates for each assay. However, including replicates could result in overfitting of the machine learning model. Before removing the replicates, their binding affinities were averaged before removal in order to have the best representation for the machine learning model to learn from. Only the first entry of the three replicates was kept. Rows with missing values were also excluded. After cleaning, there were $87,807$ usable records of the $1,259,701$ total records.


**CoV-AbDab:**
The data in the [CoV-AbDab](https://opig.stats.ox.ac.uk/webapps/covabdab/) database references both papers and patents to collect antibodies and nanobodies that are known to bind to at least one of SARS-CoV-2, SARS-CoV-1, MERS-CoV, and/or other beta-coronaviruses. Metadata such as antibody/nanobody, full variable domain sequence, origin species, structure, the binding coronavirus, and protein/epitope information are included in this database.


This data comes from a public database, and required cleaning and filtering to be most usable for the classification machine learning model. Columns of the original database were selected down to 15, keeping essential information such as sequence data of the heavy and light chain variable and complementary regions, the coronavirus that the antibody/nanobody binds to, and the origin species. SARS-CoV1, SARS-CoV2, and MERS-CoV coronaviruses were isolated, and human and mouse species were isolated as well. Data from multiple columns were standardized to fix entry errors or simplify entries. Empty and duplicate sequences were deleted. A labeling column was added based on whether or not the binding coronavirus was SARS-CoV-2 or not for usage in the classification model.


**IGMT:**


**Sab-Dab:**


## Database collection schemas


**AlphaSeq schema:**


![as_schema](mongoDB_schemas/alphaseq_schema.png)


**Example of an AlphaSeq record that fits the collection's schema:**
![alphaseq](mongoDB_schemas/alphaseq_entry_example.png)


**Example of a CoV-AbDab record that fits the collection's schema:**
![covabdab](mongoDB_schemas/covabdab_entry_example.png)


**Example of an IMGT record that fits the collection's schema:**
![imgt](mongoDB_schemas/imgt_entry_example.png)


**Example of a SAbDab record that fits the collection's schema:**
![sabdab](mongoDB_schemas/sabdab_entry_example.png)


## Classification Model
Talk about data bias, cross-reactivity


## Regression Model
Due to the AlphaSeq dataset being the only dataset with quantitative data, and because of the single chain antibody structural difference, only the AlphaSeq dataset was used for the regression model. The AlphaSeq dataset alone is sufficient for a machine learning model, with $87,807$ rows.


The data was split into $80%$ training and $20%$ testing data.
TAPES embedding was used to embed the sequences. 


Preliminary results were generated with a test file of 1000 lines, due to runtime issues when attempting to run on the full file. The hyperparameters such as number of epochs, batch size, and learning rate were fine-tuned in order to increase accuracy of the model (which is indicated by a low mean squared error). 


![alphaseq_graphs](https://user-images.githubusercontent.com/76976889/225943730-9ee469e4-99cd-4da5-ade4-f4ef3750be7f.png)
These graphs show the mean squared error before and after fine-tuning the hyperparameters. Fine-tuning resulted in a batch size of $100$ and a learning rate of $0.0001$. As the number of epochs increases, the MSE approaches $0$, indicating high accuracy. The testing dataset has higher accuracy than the training dataset, which suggests that the model is not overfit to the training data for these preliminary results. 


![2_layer](https://user-images.githubusercontent.com/76976889/225944269-5aefbb90-43da-4a6b-83b8-ac7cdfd50753.png)

Preliminary results were also created with the model with $2$ LSTM layers. The mean squared error was higher starting at $10$, and more epochs were needed ($600$) to have the mean squared error approach $0$.


To attempt to improve the runtime, the sequences were kmerized and embedded with TAPES, generating a json file. The json file serves as a reference file with the embeddings, which should improve the model runtime. However, there was unfortunately not enough time to fully implement the kmerized embeddings to run the model on the full AlphaSeq dataset. So far, the kmerized file was successfully loaded into the model along with the AlphaSeq csv. However, associating the kmerized sequences with the csv has proven to be challenging. We also discovered the kmers are generated every two nucleotides.


