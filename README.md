# KGChat
KGChat (Knowledge Graph Chat) is a chatbot designed to empower users to query and interact with knowledge graphs derived from their own event logs. With KGChat, users can uncover and analyze business processes within their data, gaining valuable insights through an intuitive and conversational interface. 

The application utilizes Large Language Models (LLMs) to process user queries and generate responses. It employs Graph Neural Network (GNN) to process and comprehend the structure of the knowledge graph. Additionally, it leverages Graph Retrieval-Augmented Generation (RAG) to ensure that contextual information is incorporated in order to generate accurate and relevant answers.

## Getting Started 

You have the option to just compile and start a docker container or create an environment and run the website directly in there.
There are a couple of config arguments that are important when starting the program. For those have a look at the [Configuration](#configuration) section.

### Requirements
The locally installed nvidia cuda driver version has to match the one in the `Dockerfile`, when using Docker, and otherwise look at it when installing cuda.

Make sure you read the whole `Getting Started` section before trying it yourself to make sure you understood everything.

Depending on the configuration, you either need a `huggingface token` or an `openai-api-key`. Both have their own config file in the `webapp/src/utils/` folder. 

### Usage with Docker

To use Docker for the execution. First make sure you have docker installed and the service is running:

1. Navigate to the `webapp` directory:
    ```sh
    $ cd webapp
    ```
2. Compile the container:
    ```sh
    $ docker build -t kgchat .
    ```
3. Start the container (the port opening is important to access the webpage):
    ```sh
    $ docker run -p 5000:5000 kgchat
    ```

### Usage without Docker

If you just want to start without Docker you first have to create an environment to run it in:

1. Navigate to the `webapp` directory:
    ```sh
    $ cd webapp
    ```
   
2. Create and activate a conda environment from the `environment.yml`:
    ```sh
    $ conda create -f environment.yml -n g-docker
    $ conda activate g-docker
    ```
3. Install cuda driver and replace `XX` at the end with the version you have installed (for example: v.11.8 = 118):
   ```sh
   $ nvcc --version # shows you the cuda version that is installed (if one is installed)
   $ conda run -n g-docker pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXX
   ```
3. Start the app:
    ```sh
    $ python app.py
    ```
## Configuration
The app can be started with different arguments. These have to be changed in the `config.txt` inside the `webapp` folder. The options are:

#### --host
- Regulates how to access the website
- Expects an IP address
- Defaults to `127.0.0.1` for local use only
- Use `0.0.0.0` for access through the host machine's IP address if the router forwards the ports 80->5000 for the host's IP

#### --mode
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

#### --llm_model
- Changes the model/repo to clone from Huggingface for the `local` use case
- Defaults to `meta-llama/Llama-2-7b-chat-hf`
- Ensure it is a text-based model
- Ensure you have enough resources to execute the model

#### --gpt_model
- Changes the model which the ChatGPT API uses for the `remote` use case
- Defaults to `gpt-3.5-turbo-16k`
- Be aware: the newer the model, the higher the cost per token
- Only use text generation models

#### --debug
- Changes whether there are debug outputs in the console
- If the flag is included, the debug messages are shown


## User Interface and Features

### Importing Event Log

After launching the application on the chosen port and accessing it on a web browser, you will arrive on the landing page. By clicking on the `Start New Chat` button (1), you can import the event log to be analyzed from your file system. It is required that the event log to be in XES format, otherwise an error will be thrown. 

<img src="User Manual/Picture1.png">

### Selecting Attributes / Columns

Once the event log is successfully imported, you will be prompted to choose the attributes, or more specifically, columns from your event log that should be included for generating the knowledge graph and be analyzed. The three attributes *concept:name*, *time:timestamp* and *case:concept:name*, which correspond to the activity name, timestamp and case ID, are selected by default since they are essential for process discovery. To pick an attribute, click on the box next to the attribute name. Once you are done, click on the `Save` button on the bottom of the page to proceed. It should be noted that at least one attribute other than the three mandatory columns has to be selected.

<img src="User Manual/2.PNG"> 

If everything runs smoothly, you will be redirected to the main page. You will find a chat session is created below the section `Chats`. By default, the chat session is given a name similar to your event log.


<img src="User Manual/4.PNG">

### Renaming and Deleting Chat Session
At any time, it is possible to rename or delete a chat session. Click on the button `...` beside a chat session and you will find the buttons for renaming and deleting it. 

<img src="User Manual/Picture5.png">

### Changing Attributes / Columns
The user is not restricted by the initial choice of the columns considered for analysis. It is also possible to select new columns or removing columns which were selected previously. This can be done by clicking on the `Select Columns` button (1). 

<img src="User Manual/Picture2.png">

To add new columns, click on the button beside the columns' names. To remove a previously selected column, uncheck the box beside the column's name. At the end, click on the `Save` button (2) to save the new selection of columns. 

<img src="User Manual/Picture3.png">

### Displaying the Full Knowledge Graph

To see the complete knowledge graph generated from the given event log, click on the `Show Full Graph` button (1). The knowledge graph will then be displayed on the rightmost section of the web page. 

<img src="User Manual/Picture4.png">

The displayed graph is not a static image, so it is possible to interact with it by clicking on the nodes or edges and moving your mouse around. On top of that, by hovering above a node in the graph, information regarding the node will be shown. 

<img src="User Manual/7.PNG">

### Chatting with the ChatBot

To pose a query regarding your process, click on the white rectangular column and type it out (1). Once you are done, click on the `Ask` button on the right and the application will process your question. Our application does not only support processing questions in English but also in other languages. We have tried posing questions in German, Chinese, Spanish and Korean, and have received satisfied answers. 

<img src="User Manual/Picture6.png">

The questions and the corresponding generated answers will be displayed above the text box. To see the subgraph which is contextually relevant to the question, click on the button `show subgraph` below the conversation (3). For every question, there will be a corresponding contextually-relevant subgraph generated and it can be accessed at any time with the aforementioned button. 

<img src="User Manual/Picture7.png">

After clicking on the `show subgraph` button, the subgraph will be displayed on the right side of the conversation section. Similar to the full graph, the subgraph is also interactive. Above the subgraph is the corresponding user's question that is contextually related to the subgraph.  

<img src="User Manual/11.PNG">

### Starting New Chat Session

To start on a new chat session, click on the `Start New Chat` button and the following procedure is same as importing a new event log. A new chat session will then be created and is accessible within the Chats section of the web page. 

<img src="User Manual/Picture8.PNG">

## Troubleshooting
## FAQ
## Glossary and Index