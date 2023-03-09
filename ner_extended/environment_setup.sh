mkdir environment
python3 -m venv environment/training_env
source environment/training_env/bin/activate
pip install -r "requirements.txt"
pip install spacy[cuda101]
huggingface-cli login
wandb login