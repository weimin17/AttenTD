
This is the code repository of the NAACL 2022 paper [A Study of the Attention Abnormality in Trojaned BERTs](https://aclanthology.org/2022.naacl-main.348/). It proposes an attention-based Trojan detector to distinguish Trojaned models from clean ones (in the field of Natural Language Processing). 



# Setup Environment

~~~~
bash setup_python_environment.sh
~~~~

to setup environment.


# AttenTD Detector

You can simply feed your suspicious model into detector, and output the probability of whether the model is trojaned or not.



In order to run the detector, please simply run 


~~~~
python detection_attentd.py
~~~~

Please model path and setting accordingly:

1. Be sure to change your model file path 'model_filepath' in file `detection_attentd.py` line 470.


2. The suspicious models trained with different codebases may be different, so please BE SURE that:

    1) Pleae be sure to adjust the model inference code to your own case [in function gene_batch_logits in file `detection_attentd.py`]. 

    2) Please check the tokenizer [in function baseline_trigger_reconstruction in file `detection_attentd.py`].



Some additional information (you do not need to do anything for those information, just the default setting should be fine.)

a. The pre defined trigger candidate set is in `./pre_defined_data/trigger_hub.pkl`, clean sentence samples are stored in `./pre_defined_data/dev-custom-imdb`. 

b. The output log and file would be stored in `./results/cls_data` and `./results/cls_results`

