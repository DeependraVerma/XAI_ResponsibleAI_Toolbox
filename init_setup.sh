echo "Initializing setup at $(date)"
conda create -p venv python=3.8 -y

conda activate venv

pip install -r requirements_dev.txt

pip install -e .
echo "Setup completed at $(date)"