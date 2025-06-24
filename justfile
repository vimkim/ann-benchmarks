uv-init:
    uv venv --python 3.10.6

uv-source-nu:
    overlay use .venv/bin/activate.nu

uv-sync:
    uv pip install -r requirements.txt
