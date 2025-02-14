# llm_building_blocks

In this repository I plan to implement some LLM libraries / algorithms from scratch. Some of the implementations include:

1) Tokenizers
2) Attention Block
3) GPT-style Language Model
4) PEFT
5) RLHF

## Setting up the Environment:

To clone this repository, open a terminal and run:

`git clone https://github.com/micheldc55/llm_building_blocks.git`

How to install and set up the environment using `uv`:

If you are working behind a proxy / SSL certificate, export the path to the certificate to all necessary channels:

1) `export SSL_CERT_FILE=</path/to/zscaler.pem>`
2) `export REQUESTS_CA_BUNDLE=</path/to/zscaler.pem>`
3) `export CURL_CA_BUNDLE=</path/to/zscaler.pem>`
4) ....

In order to make this changes persistent, you can also open a terminal and run the following command:

```bash
echo 'export SSL_CERT_FILE=</path/to/zscaler.pem>' >> ~/.bashrc
echo 'export REQUESTS_CA_BUNDLE=</path/to/zscaler.pem>' >> ~/.bashrc
echo 'export CURL_CA_BUNDLE=</path/to/zscaler.pem>' >> ~/.bashrc
source ~/.bashrc
```

To create (and activate) the virtual environment using uv, open a terminal and run:

```bash
uv venv
source .venv/bin/activate
```

If you are **not behind a proxy**, install dependencies with:

```bash
uv pip install -r <(uv pip compile pyproject.toml)
```

However, **if you are behind a proxy or need custom SSL settings**, use the --native-tls flag:

```bash
uv pip install -r <(uv pip compile pyproject.toml) --native-tls
```