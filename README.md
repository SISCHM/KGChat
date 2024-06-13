# KGChat
Creating a ChatBot to ease up Process Discovery Using Python



- The app can be started with different arguments. These have to be changed in the 
    config.txt inside the webapp folder. The options are:
  - --host
      - regulates how to access the website
      - expects an ip address
      - defaults to 127.0.0.1 for local use only
      - use 0.0.0.0 for access through the host machines ip address, if the router forwards the ports 80->5000 for the hosts ip
  - --llm_mode
      - changes which LLM type is used
      - expects either "local" (default) or "remote"
        - local: uses a Huggingface Model which will be downloaded and run locally on your machine
          - This requires a lot of power
          - A Huggingface Access Token is required in the corresponding file in /webapp/src/utils/HUGGINGFACE_TOKEN.txt
          - Make sure the token allows access to repos that are not yours
          - Make sure you have requested (and granted) access to the Model you want to use
      - remote: uses the ChatGPT api to run the LLM remote
        -  A ChatGPTApi Key is required in the corresponding file in /webapp/src/utils/OPENAI_API_KEY.txt
        - Make sure you have topped up the balance, otherwise the request will return an error
  - --llm_model
      - changes the model/repo to clone from Huggingface for the "local" use case
      - defaults to meta-llama/Llama-2-7b-chat-hf
      - make sure it is a text-based model
      - make sure you have enough resources to execute the model
  - --gpt_model
      - changes the model which the ChatGPT Api uses for the "remote" use case
      - defaults to gpt-3.5-turbo-16k
      - be aware: the newer the model the higher the cost per Token
      - only use text generation models
  - --debug
      - changes whether there are debug outputs in the console
      - if the flag is included, the debug messages are shown
    

- To use Docker for the execution:
    - go to webapp <br>
      $ cd webapp <br>
    - Then compile container <br>
      $ docker build -t kgchat . <br>
    - Then start the container using <br>
      $ docker run -p 5000:5000 kgchat

- If you just want to start without docker:
    - go to webapp <br>
      $ cd webapp <br>
    - Then start the app <br>
      $ python app.py
  
