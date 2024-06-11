# KGChat
Creating a ChatBot to ease up Process Discovery Using Python

To use Docker go to webapp
$ cd webapp
Then compile container
$ docker build -t kgchat .
Then start the container using
$ docker run -p 5000:5000 kgchat

If you just want to start without docker go to webapp
$ cd webapp
Then start the app
$ python app.py <-- for local and ip 127.0.0.1:5000
$ python app.py 0.0.0.0 <-- for loading on your ip:5000 
=====
