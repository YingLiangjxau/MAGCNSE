# MAGCNSE
This is the implementation for our paper:
>MAGCNSE: predicting lncRNA-disease associations using multi-view attention graph convolutional network and stacking ensemble model
## File Description 
>param.py: assign values to hyperparameters used in this study.  
dataprocessing.py: the code for data processing.  
MAGCN.py: the code for model architecture in representation learning module.  
representaionlearning.py: obtain the representations of lncRNAs and diseases.  
concat.py: get the representations of positive and negative lncRNA-disease pairs.  
stackingensemble.py: an example to get the prediction results by stacking ensemble model. All machine learning classifiers in this file use default parameters, the readers can use grid search to find optimal parameters for each machine learning classifier according to their own data. 

## Usage
Enviroment:
>python==3.7  
pytorch==1.5.1  
torch-geometric==1.6.0  
 
## Data
The lncRNA-disease associations are collected from LncRNADisease v2.0(http://www.rnanut.net/lncrnadisease/), the DOIDs of diseases are obtained from Disease Ontology(https://disease-ontology.org/), the sequences of lncRNAs are obtained from NONCODE v6.0(http://www.noncode.org/).
