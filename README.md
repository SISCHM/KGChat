# KGChat
Creating a ChatBot to ease up Process Discovery Using Python

- need huggingface token
    - make sure it is allowed to access public repos you have access to
    - need access to meta-llama/Llama-2-7b-chat-hf
    
- To use Docker go to webapp <br>
    - $ cd webapp <br>
    - Then compile container <br>
    - $ docker build -t kgchat . <br>
    - Then start the container using <br>
    - $ docker run -p 5000:5000 kgchat

- If you just want to start without docker go to webapp <br>
    - $ cd webapp <br>
    - Then start the app <br>
    - $ python app.py <-- for local and ip 127.0.0.1:5000 <br>
    - $ python app.py 0.0.0.0 <-- for loading on your ip:5000
