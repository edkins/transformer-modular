```
python3 -m venv env
source env/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python modular.py train
python modular.py final embed
```
