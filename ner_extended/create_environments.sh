rm -rf environments/
mkdir -p/environments

python3 -m venv environments/preprocessing_env
python3 -m venv environments/training_env
python3 -m venv environments/packaging_env

source environments/packaging_env/bin/activate
pip install -r "requirements_preprocessing_env.txt"
deactivate

source environments/training_env/bin/activate
pip install -r "requirements_training_env.txt"
pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install spacy[cuda101]
pip install --upgrade torch
wandb login # insert API-key from https://wandb.ai/settings
deactivate

source environments/packaging_env/bin/activate
pip install -r "requirements_packaging_env.txt"
pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install spacy[cuda101]
pip install --upgrade torch
huggingface-cli login # insert token (WRITE) from https://huggingface.co/settings/tokens
deactivate