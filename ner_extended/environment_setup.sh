mkdir environment
python3 -m venv environment/training_env
source environment/training_env/bin/activate
pip install wheel==0.38.4 # no version works
pip install numpy==1.23.3
pip install spacy==3.5.0 # no version works
#pip install spacy-transformers # below version has dependency of transformers that matches the dependency of spacy-huggingface-hub
pip install spacy-transformers==1.1.2
pip install torch==1.13.1 # no version works
pip install spacy[cuda101] # no idea about version? but version from 30 Jan 2023 works
pip install huggingface==0.0.1 # no version works
pip install spacy-huggingface-hub==0.0.8 #no version works
pip install wandb==0.13.9 # no version works
wandb login