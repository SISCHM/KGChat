# KGChat
Creating a ChatBot to ease up Process Discovery Using Python

## Configuration

The app can be started with different arguments. These have to be changed in the `config.txt` inside the `webapp` folder. The options are:

### --host
- Regulates how to access the website
- Expects an IP address
- Defaults to `127.0.0.1` for local use only
- Use `0.0.0.0` for access through the host machine's IP address if the router forwards the ports 80->5000 for the host's IP

### --mode
- Changes which LLM type is used
- Expects either `local` (default) or `remote`
  - `local`: Uses a Huggingface Model which will be downloaded and run locally on your machine
    - This requires a lot of power
    - A Huggingface Access Token is required in the corresponding file in `/webapp/src/utils/HUGGINGFACE_TOKEN.txt`
    - Ensure the token allows access to repos that are not yours
    - Ensure you have requested (and been granted) access to the Model you want to use
  - `remote`: Uses the ChatGPT API to run the LLM remotely
    - A ChatGPT API Key is required in the corresponding file in `/webapp/src/utils/OPENAI_API_KEY.txt`
    - Ensure you have topped up the balance, otherwise, the request will return an error

### --llm_model
- Changes the model/repo to clone from Huggingface for the `local` use case
- Defaults to `meta-llama/Llama-2-7b-chat-hf`
- Ensure it is a text-based model
- Ensure you have enough resources to execute the model

### --gpt_model
- Changes the model which the ChatGPT API uses for the `remote` use case
- Defaults to `gpt-3.5-turbo-16k`
- Be aware: the newer the model, the higher the cost per token
- Only use text generation models

### --debug
- Changes whether there are debug outputs in the console
- If the flag is included, the debug messages are shown

## Usage with Docker

To use Docker for the execution:

1. Navigate to the `webapp` directory:
    ```sh
    $ cd webapp
    ```
2. Compile the container:
    ```sh
    $ docker build -t kgchat .
    ```
3. Start the container:
    ```sh
    $ docker run -p 5000:5000 kgchat
    ```

## Usage without Docker

If you just want to start without Docker:

1. Navigate to the `webapp` directory:
    ```sh
    $ cd webapp
    ```
2. Start the app:
    ```sh
    $ python app.py
    ```
