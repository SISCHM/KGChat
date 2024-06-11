from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from threading import Lock
import os
import json
import pickle
import sys
import shutil
from src import EventLog
from src import Conversation
from src import TextEmbedder
from src import Graph

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'chats'
app.config['ALLOWED_EXTENSIONS'] = {'xes'}
question_lock = Lock()

def load_selected_columns(chat_id):
    selected_columns_file = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, 'selected_columns.json')
    if os.path.exists(selected_columns_file):
        with open(selected_columns_file, 'r') as f:
            return json.load(f)
    return []

def load_all_columns(chat_id):
    selected_columns_file = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, 'all_columns.json')
    if os.path.exists(selected_columns_file):
        with open(selected_columns_file, 'r') as f:
            return json.load(f)
    return []

def save_graph(graph, save_path):
    with open(save_path, 'wb') as file:
        pickle.dump(graph, file)
    '''    
    if os.path.exists(save_path):
        print(f"Graph successfully saved at {save_path}")
    else:
        print(f"Failed to save graph at {save_path}")
    '''

def load_graph(load_path):
    with open(load_path, 'rb') as file:
        graph = pickle.load(file)
    return graph

def load_conv_from_pickle(chat_id):
    with open(f"{app.config['UPLOAD_FOLDER']}/{chat_id}/conversation.pkl", 'rb') as f:
        return pickle.load(f)

def save_conv_to_pickle(chat_id, conversation):
    with open(f"{app.config['UPLOAD_FOLDER']}/{chat_id}/conversation.pkl", 'wb') as f:
        pickle.dump(conversation, f)

def save_text_embedder(embedder, filename):
    with open(f"{app.config['UPLOAD_FOLDER']}/{filename}/embedder.pkl", 'wb') as f:
        pickle.dump(embedder, f)

def load_text_embedder(filename):
    with open(f"{app.config['UPLOAD_FOLDER']}/{filename}/embedder.pkl", 'rb') as f:
        return pickle.load(f)

def prepare_log(filename):
    event_log = EventLog.EventLog(f"{app.config['UPLOAD_FOLDER']}/{filename}/{filename}.xes")
    all_columns = event_log.log.columns.tolist()
    with open(f"{app.config['UPLOAD_FOLDER']}/{filename}/all_columns.json", 'w') as f:
        json.dump(all_columns, f)
    return event_log

def preprocesslog(event_log, selected_columns, chat_id):
    event_log.preprocess_log(selected_columns)
    know_g = event_log.create_kg()
    save_graph(know_g,os.path.join(app.config['UPLOAD_FOLDER'], chat_id, "graphs", f"0.pkl"))
    embedder = TextEmbedder.TextEmbedder()
    know_g.embed_graph(event_log.name, embedder)
    save_text_embedder(embedder, event_log.name)
    conversation = Conversation.Conversation(know_g)# , "UnderstandLing/llama-2-3b-chat-nl-lora")
    save_conv_to_pickle(event_log.name, conversation)

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
            conv = json.load(f)
            for key, value in conv.items():
                value['answer'] = value['answer'].replace('\n', '<br>')
            return conv
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

    graph_url = url_for('show_graph', chat_id=chat_id, graph_index=0)
    all_columns = load_all_columns(chat_id)
    selected_columns = load_selected_columns(chat_id)

    return render_template('chat.html', chat_id=chat_id, chat_name=chat_name, conversation=conversation, chats=chats, graph_url=graph_url, graph_title="Full Graph", columns=all_columns, selected_columns=selected_columns)

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
        all_columns = EventLog.extract_columns_from_xes(os.path.join(app.config['UPLOAD_FOLDER'], filename, filename + ".xes"))
        return render_template('select_columns.html', chat_id=filename, columns=all_columns)
    return redirect(url_for('home'))

@app.route('/select_columns/<chat_id>', methods=['GET', 'POST'])
def select_columns(chat_id):
    if request.method == 'POST':
        selected_columns = request.form.getlist('columns')
        if not selected_columns:
            return "You must select at least one column", 400
        always_selected = ["concept:name", "time:timestamp", "case:concept:name"]
        selected_columns.extend(always_selected)
        selected_columns = list(set(selected_columns))  # Remove duplicates
        with open(f"{app.config['UPLOAD_FOLDER']}/{chat_id}/selected_columns.json", 'w') as f:
            json.dump(selected_columns, f)
        event_log = prepare_log(chat_id)
        preprocesslog(event_log, selected_columns, chat_id)
        return redirect(url_for('chat', chat_id=chat_id))

@app.route('/save_columns/<chat_id>', methods=['POST'])
def save_columns(chat_id):
    selected_columns = request.form.getlist('columns')
    always_selected = ["concept:name", "time:timestamp", "case:concept:name"]
    selected_columns.extend(always_selected)
    selected_columns = list(set(selected_columns))  # Remove duplicates
    if not selected_columns:
        return "You must select at least one column", 400
    with open(f"{app.config['UPLOAD_FOLDER']}/{chat_id}/selected_columns.json", 'w') as f:
        json.dump(selected_columns, f)
    event_log = prepare_log(chat_id)
    preprocesslog(event_log, selected_columns, chat_id)
    return redirect(url_for('chat', chat_id=chat_id))

@app.route('/show_graph/<chat_id>/<graph_index>')
def show_graph(chat_id, graph_index):
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, "graphs", f"{graph_index}.pkl")
    if os.path.exists(graph_path):
        graph = load_graph(graph_path)
    else:
        print(f"Graph file not found at: {graph_path}")
        return jsonify({'error': 'Graph file not found'}), 404

    nodes = [{'id': idx, 'label': row['node_name']} for idx, row in graph.nodes.iterrows()]
    edges = [{'from': row['Source_id'], 'to': row['Destination_id'], 'Frequency': row['Frequency'],
              'Average_time': row['Average_time']} for idx, row in graph.edges.iterrows()]

    graph_title = "Full Graph" if int(graph_index) == 0 else \
    list(load_conversation(chat_id).values())[int(graph_index) - 1]['question']
    return jsonify({'nodes': nodes, 'edges': edges, 'graph_title': graph_title})

def ask_question(chat_id, question):
    conv = load_conv_from_pickle(chat_id)
    embedder = load_text_embedder(chat_id)
    subgraph = conv.know_g.retrieve_subgraph_pcst(question, embedder)
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], chat_id, "graphs", f"{len(conv.prev_conv)+1}.pkl")
    save_graph(subgraph, graph_path)
    Conversation.check_available_ram()
    conv.ask_question(subgraph, question)
    conv.question_to_file(f"{app.config['UPLOAD_FOLDER']}/{chat_id}", question)
    save_conv_to_pickle(chat_id, conv)
    save_text_embedder(embedder, chat_id)

@app.route('/ask_question/<chat_id>', methods=['POST'])
def handle_ask_question(chat_id):
    question = request.form['question']
    if question_lock.locked():
        return jsonify({'error': "Another question is being processes. Please wait."}), 423

    with question_lock:
        ask_question(chat_id, question)

    return redirect(url_for('chat', chat_id=chat_id))

@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    chat_path = os.path.join(app.config["UPLOAD_FOLDER"], chat_id)
    if os.path.isdir(chat_path):
        shutil.rmtree(chat_path)
    return jsonify({'success': True})

@app.route('/rename_chat/<chat_id>', methods=['POST'])
def rename_chat(chat_id):
    chat_path = os.path.join(app.config['UPLOAD_FOLDER'], chat_id)
    if os.path.isdir(chat_path):
        data = request.get_json()
        new_name = data['name'].strip()
        info_file = os.path.join(chat_path, 'info.txt')
        with open(info_file, 'w') as f:
            f.write(new_name)
    chats = get_chats()
    return render_template('index.html', chats=chats)

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        app.run(debug=True)
    else:
        app.run(debug=True, host=sys.argv[1])

