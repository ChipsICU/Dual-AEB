# Usage

## Environment

- STEP 1: Create enviroment
    ```
    conda create -n dual_aeb python=3.8 -y
    conda activate dual_aeb

    pip install --upgrade pip  # Enable PEP 660 support.
    ```
- STEP 2: Install cudatoolkit

    ```
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```

- STEP 3: Install torch
    ```
    conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    or
    ```
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    ```

- STEP 4: Install LLAVA requirements

    ```
    cd /path/to/LLava
    pip install -e ".[train]"
    ```

- STEP 5: Install evaluation packages

    ```
    pip install nltk rouge-score pycocoevalcap prettytable
    cd /path/to/root
    mkdir nltk_data
    wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip
    mkdir chunkers grammars misc sentiment taggers corpora help models stemmers tokenizers
    unzip wordnet.zip -d /path/to/root/nltk_data/corpora
    ```

- STEP 6: Install flash-attn

    ```
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
    pip install flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
    ```

## Training

```
./scripts/train/finetune_onevision.sh
```