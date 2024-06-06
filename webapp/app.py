from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import io
import base64
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'chats'
app.config['ALLOWED_EXTENSIONS'] = {'xes'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_chats():
    chats = []
    for chat_folder in os.listdir(app.config['UPLOAD_FOLDER']):
        chat_path = os.path.join(app.config['UPLOAD_FOLDER'], chat_folder)
        if os.path.isdir(chat_path):
            chat_name = chat_folder
            info_file = os.path.join(chat_path, 'info.txt')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    chat_name = f.read().strip()
            chats.append({'id': chat_folder, 'name': chat_name})
    return chats

def load_conversation(chat_id):
    conv_file = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, 'conv.json')
    if os.path.exists(conv_file):
        with open(conv_file, 'r') as f:
            return json.load(f)
    return {}

@app.route('/')
def home():
    chats = get_chats()
    return render_template('index.html', chats=chats)

@app.route('/chat/<chat_id>')
def chat(chat_id):
    chats = get_chats()
    conversation = load_conversation(chat_id)

    chat_name = chat_id
    info_file = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, 'info.txt')
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            chat_name = f.read().strip()

    graph_url = url_for('get_graph', chat_id=chat_id, graph_index=0)

    return render_template('chat.html', chat_id=chat_id, chat_name=chat_name, conversation=conversation, chats=chats, graph_url=graph_url, graph_title="Full Graph")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    if file and allowed_file(file.filename):
        filename = file.filename.rsplit('.', 1)[0]
        new_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(new_folder_path, exist_ok=True)
        file.save(os.path.join(new_folder_path, file.filename))
        return redirect(url_for('chat', chat_id=filename))
    return redirect(url_for('home'))

@app.route('/show_graph/<chat_id>/<graph_index>')
def show_graph(chat_id, graph_index):
    conversation = load_conversation(chat_id)
    graph_index = int(graph_index)
    if graph_index == 0:
        graph_title = "Full Graph"
        graph_url = url_for('get_graph', chat_id=chat_id, graph_index=0)
    else:
        message = list(conversation.values())[graph_index - 1]
        graph_title = message['question']
        graph_url = url_for('get_graph', chat_id=chat_id, graph_index=graph_index)
    return jsonify(graph_url=graph_url, graph_title=graph_title)

@app.route('/get_graph/<chat_id>/<graph_index>')
def get_graph(chat_id, graph_index):
    graph_index = int(graph_index)
    graph_file = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, f'graphs/{graph_index}.png')
    print(graph_file)
    if os.path.exists(graph_file):
        return send_file(graph_file, mimetype='image/png')
    return "", 404

def ask_question(chat_id, question):
    # Placeholder for actual question processing
    print(f"Question asked: {question}")

    # Add the question to the conversation (placeholder logic)
    conversation = load_conversation(chat_id)
    conversation[str(len(conversation) + 1)] = {
        "question": question,
        "sub_graph": "Example sub graph",
        "answer": "This is a placeholder answer",
        "graph": {"x": [1, 2, 3], "y": [4, 5, 6]}
    }

    # Save the updated conversation
    conv_file = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, 'conv.json')
    with open(conv_file, 'w') as f:
        json.dump(conversation, f)

@app.route('/ask_question/<chat_id>', methods=['POST'])
def handle_ask_question(chat_id):
    question = request.form['question']
    ask_question(chat_id, question)
    return redirect(url_for('chat', chat_id=chat_id))

if __name__ == '__main__':
    app.run(debug=True)
