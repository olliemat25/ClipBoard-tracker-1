import tkinter as tk
from tkinter import ttk, messagebox
import pyperclip
from pynput import keyboard
from tinydb import TinyDB, Query
from transformers import pipeline
import time
import threading
import queue
import datetime
import traceback
import os

# -------------------------
# AI CLASSIFIER (async)
# -------------------------
classifier = None
zero_shot = None
embedder = None
classifier_ready = threading.Event()
embed_ready = threading.Event()
classification_queue = queue.Queue()
db_lock = threading.Lock()

# broad topic labels used for zero-shot topic classification
BROAD_LABELS = [
    "Formal Sciences", "Physical Sciences", "Life Sciences", "Applied Sciences", "Social Sciences", "Humanities", "Professional Disciplines", "Arts & Media", "AI", "Research", "Programming", "Math", "School", "Cooking", "Other"
]
SUB_LABELS_BY_BROAD = {
    "Formal Sciences": ["Logic", "Mathematics", "Statistics"],
    "Physical Sciences": ["Astronomy", "Chemistry", "Geology", "Physics"],
    "Life Sciences": ["Biology", "Botany", "Ecology", "Neuroscience", "Zoology"],
    "Applied Sciences": ["Agriculture", "Architecture", "Computer Science", "Engineering", "Health Sciences", "Medicine"],
    "Social Sciences": ["Anthropology", "Archaeology", "Criminology", "Economics", "Geography", "International Relations", "Political Science", "Psychology", "Sociology"],
    "Humanities": ["Art History", "Classics", "History", "Linguistics", "Literature", "Philosophy", "Religious Studies"],
    "Professional Disciplines": ["Business", "Education", "Finance", "Law", "Management", "Marketing"],
    "Arts & Media": ["Design", "Film", "Journalism", "Music", "Theater", "Visual Arts"],
}
ZS_CANDIDATE_LABELS = BROAD_LABELS + [
    "Logic", "Mathematics", "Statistics", "Astronomy", "Chemistry", "Geology", "Physics",
    "Biology", "Botany", "Ecology", "Neuroscience", "Zoology", "Agriculture", "Architecture",
    "Computer Science", "Engineering", "Health Sciences", "Medicine", "Anthropology", "Archaeology",
    "Criminology", "Economics", "Geography", "International Relations", "Political Science", "Psychology",
    "Sociology", "Art History", "Classics", "History", "Linguistics", "Literature", "Philosophy",
    "Religious Studies", "Business", "Education", "Finance", "Law", "Management", "Marketing",
    "Design", "Film", "Journalism", "Music", "Theater", "Visual Arts",
    "AI", "Machine Learning", "Research", "Programming", "Cooking", "School", "Math", "Other"
]

LABEL_TEXTS = {
    "Formal Sciences": "Formal sciences cover logic, mathematics, and statistics.",
    "Physical Sciences": "Physical sciences include astronomy, chemistry, geology, and physics.",
    "Life Sciences": "Life sciences cover biology, botany, ecology, neuroscience, and zoology.",
    "Applied Sciences": "Applied sciences include agriculture, architecture, computer science, engineering, health sciences, and medicine.",
    "Social Sciences": "Social sciences include anthropology, archaeology, criminology, economics, geography, international relations, political science, psychology, and sociology.",
    "Humanities": "Humanities cover art history, classics, history, linguistics, literature, philosophy, and religious studies.",
    "Professional Disciplines": "Professional disciplines include business, education, finance, law, management, and marketing.",
    "Arts & Media": "Arts & Media cover design, film, journalism, music, theater, and visual arts.",
    "AI": "Artificial intelligence and machine learning techniques, models, and research.",
    "Research": "Academic research including studies, papers, experiments, analysis, and benchmarks.",
    "Programming": "Computer programming and software development topics.",
    "Math": "General mathematics and related concepts.",
    "School": "School-related and homework topics.",
    "Cooking": "Cooking, recipes, and food preparation.",
}
for label in ZS_CANDIDATE_LABELS:
    LABEL_TEXTS.setdefault(label, label)

LABEL_EMBED = None

import math

def get_embedding(text):
    if embedder is None:
        return None
    try:
        out = embedder(text)
        if isinstance(out, list) and len(out) > 0:
            arr = out[0] if isinstance(out[0], list) and isinstance(out[0][0], list) else out
            if len(arr) and isinstance(arr[0], list):
                vec = [sum(col) / len(arr) for col in zip(*arr)]
            else:
                vec = arr
            return vec
    except Exception:
        traceback.print_exc()
    return None


def cosine_similarity(u, v):
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(x * y for x, y in zip(u, v))
    norm_u = math.sqrt(sum(x * x for x in u))
    norm_v = math.sqrt(sum(y * y for y in v))
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot / (norm_u * norm_v)


def compute_label_embeddings():
    global LABEL_EMBED
    if embedder is None:
        return
    label_embeddings = {}
    for label, text in LABEL_TEXTS.items():
        vec = get_embedding(text)
        if vec is not None:
            label_embeddings[label] = vec
    LABEL_EMBED = label_embeddings


def get_topic_from_embeddings(text):
    if embed_ready.is_set() and LABEL_EMBED:
        vec = get_embedding(text)
        if vec is None:
            return None
        best_label = None
        best_score = 0.0
        for label, lab_vec in LABEL_EMBED.items():
            score = cosine_similarity(vec, lab_vec)
            if score > best_score:
                best_score = score
                best_label = label
        if best_score > 0.30:
            return best_label
    return None


def get_topic_from_zero_shot(text):
    if zero_shot is None:
        return None
    try:
        res = zero_shot(text, candidate_labels=BROAD_LABELS)
        if res and 'labels' in res and res['scores']:
            best_label = res['labels'][0]
            best_score = res['scores'][0]
            if best_score > 0.25:
                if best_label in SUB_LABELS_BY_BROAD:
                    sub_res = zero_shot(text, candidate_labels=SUB_LABELS_BY_BROAD[best_label])
                    if sub_res and 'labels' in sub_res and sub_res['scores'] and sub_res['scores'][0] > 0.2:
                        return sub_res['labels'][0]
                return best_label
    except Exception:
        traceback.print_exc()
    return None


def get_cluster_topic(text):
    if zero_shot is not None:
        topic = get_topic_from_zero_shot(text)
        if topic:
            return topic
        try:
            res = zero_shot(text, candidate_labels=ZS_CANDIDATE_LABELS)
            if res and 'labels' in res and res['scores'] and res['scores'][0] > 0.20:
                return res['labels'][0]
        except Exception:
            traceback.print_exc()
    if embed_ready.is_set():
        topic = get_topic_from_embeddings(text)
        if topic:
            return topic
    return None


def load_classifier():
    global classifier
    try:
        # primary sentiment/classifier replaced by zero-shot topic classifier
        try:
            zero = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            globals()['zero_shot'] = zero
            print("Zero-shot classifier loaded")
        except Exception as e:
            print("Zero-shot load failed:", e)
            globals()['zero_shot'] = None

        # load compact sentence embedding model for semantic grouping
        try:
            emb = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
            globals()['embedder'] = emb
            compute_label_embeddings()
            embed_ready.set()
            print("Embedder loaded")
        except Exception as e:
            print("Embedder load failed:", e)
            globals()['embedder'] = None

        # keep a fallback basic classifier (optional)
        classifier = pipeline("text-classification")
        print("Text classifier loaded")
    except Exception as e:
        print("Failed to load classifier:", e)
        classifier = None
    finally:
        classifier_ready.set()

def categorize_heuristic(text):
    t = text.lower()
    if any(token in t for token in ["python", "code", "script", "compile", "debug", "programming"]):
        return "Computer Science"
    if any(token in t for token in ["homework", "assignment", "essay", "study"]):
        return "School"
    if any(token in t for token in ["recipe", "cook", "bake", "ingredient", "oven"]):
        return "Cooking"
    if any(token in t for token in ["logic", "proof", "theorem", "mathematics", "algebra", "geometry", "statistics", "calculus"]):
        return "Formal Sciences"
    if any(token in t for token in ["astronomy", "chemistry", "geology", "physics", "quantum", "particle", "relativity"]):
        return "Physical Sciences"
    if any(token in t for token in ["biology", "botany", "ecology", "neuroscience", "zoology", "genetics", "evolution"]):
        return "Life Sciences"
    if any(token in t for token in ["agriculture", "architecture", "engineering", "medicine", "health", "technology"]):
        return "Applied Sciences"
    if any(token in t for token in ["anthropology", "archaeology", "criminology", "economics", "geography", "political", "psychology", "sociology"]):
        return "Social Sciences"
    if any(token in t for token in ["art history", "classics", "history", "linguistics", "literature", "philosophy", "religious"]):
        return "Humanities"
    if any(token in t for token in ["business", "education", "finance", "law", "management", "marketing"]):
        return "Professional Disciplines"
    if any(token in t for token in ["design", "film", "journalism", "music", "theater", "visual arts"]):
        return "Arts & Media"
    if any(token in t for token in ["ai", "artificial intelligence", "machine learning", "ml", "neural network", "transformer", "gpt", "llm"]):
        return "AI"
    if any(token in t for token in ["research", "study", "experiment", "paper", "survey", "analysis", "benchmark"]):
        return "Research"
    return None

def categorize(text):
    # quick heuristics first
    h = categorize_heuristic(text)
    if h:
        return h
    if zero_shot is not None:
        topic = get_topic_from_zero_shot(text)
        if topic:
            return topic
    if embed_ready.is_set():
        topic = get_topic_from_embeddings(text)
        if topic:
            return topic
    if classifier_ready.is_set() and zero_shot is not None:
        try:
            res = zero_shot(text, candidate_labels=ZS_CANDIDATE_LABELS)
            if res and 'labels' in res and res['scores']:
                best_label = res['labels'][0]
                best_score = res['scores'][0]
                if best_score > 0.2:
                    return best_label
        except Exception:
            traceback.print_exc()
    if classifier_ready.is_set() and classifier is not None:
        try:
            out = classifier(text)
            if out and isinstance(out, list):
                label = out[0].get("label", "Other")
                if label not in {"POSITIVE", "NEGATIVE", "LABEL_0", "LABEL_1"}:
                    return label
        except Exception:
            traceback.print_exc()
            return "Other"
    return "Other"

def classification_worker():
    while True:
        doc_id, text = classification_queue.get()
        try:
            classifier_ready.wait(timeout=10)
            label = categorize(text)
            if not label:
                label = "Other"
            with db_lock:
                try:
                    db.update({"category": label}, doc_ids=[doc_id])
                except Exception:
                    pass
            try:
                root.after(0, update_gui)
            except Exception:
                pass
        except Exception:
            traceback.print_exc()
        finally:
            classification_queue.task_done()

# start classifier loader and worker
threading.Thread(target=load_classifier, daemon=True).start()
threading.Thread(target=classification_worker, daemon=True).start()

# -------------------------
# DATABASE
# -------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "paste_history.json")
DATA_PATH = os.path.abspath(DATA_PATH)
db = TinyDB(DATA_PATH)

last_saved_text = None

def save_paste(text=None):
    global last_saved_text
    if text is None:
        try:
            text = pyperclip.paste().strip()
        except Exception:
            return
    if not text:
        return
    if last_saved_text and text == last_saved_text:
        return
    last_saved_text = text
    category = categorize(text) or "Other"
    entry = {"text": text, "category": category, "time": time.time()}
    with db_lock:
        try:
            doc_id = db.insert(entry)
        except Exception:
            doc_id = db.insert(entry)
        try:
            db.update({'_doc_id': doc_id}, doc_ids=[doc_id])
        except Exception:
            pass
    # queue for more accurate classification
    classification_queue.put((doc_id, text))
    try:
        root.after(0, update_gui)
    except Exception:
        pass
    print(f"[Saved] {category}: {text[:40]}...")

# -------------------------
# PASTE DETECTOR
# -------------------------
pressed = set()

def on_press(key):
    try:
        pressed.add(key)
        if keyboard.Key.ctrl_l in pressed or keyboard.Key.ctrl_r in pressed:
            if hasattr(key, 'char') and key.char == 'v':
                # slight delay to let clipboard update
                time.sleep(0.05)
                save_paste()
    except Exception:
        pass

def on_release(key):
    try:
        pressed.discard(key)
    except Exception:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# Clipboard polling as backup (detect programmatic clipboard changes)
def clipboard_poller():
    prev = None
    while True:
        try:
            cur = pyperclip.paste()
            if cur is not None:
                cur = cur.strip()
            if cur and cur != prev:
                prev = cur
                save_paste(cur)
        except Exception:
            pass
        time.sleep(0.5)

threading.Thread(target=clipboard_poller, daemon=True).start()

# -------------------------
# TKINTER GUI
# -------------------------
root = tk.Tk()
root.title("Context Clipboard")
root.geometry("900x600")

top_frame = ttk.Frame(root)
top_frame.pack(fill="x")

btn_delete = ttk.Button(top_frame, text="Delete Selected")
btn_delete.pack(side="left", padx=4, pady=4)

btn_clear = ttk.Button(top_frame, text="Clear All")
btn_clear.pack(side="left", padx=4, pady=4)

sort_var = tk.StringVar(value="Newest")
sort_menu = ttk.OptionMenu(top_frame, sort_var, "Newest", "Newest", "Oldest", "Category")
sort_menu.pack(side="left", padx=4)

filter_var = tk.StringVar()
filter_entry = ttk.Entry(top_frame, textvariable=filter_var)
filter_entry.pack(side="left", padx=4)

btn_filter = ttk.Button(top_frame, text="Apply Filter")
btn_filter.pack(side="left", padx=4)

status_label = ttk.Label(top_frame, text="Classifier: Loading...")
status_label.pack(side="right", padx=8)

frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)

scrollbar = ttk.Scrollbar(frame)
scrollbar.pack(side="right", fill="y")

# Replace list-only UI with a two-pane view: list on left, detail on right
paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
paned.pack(fill='both', expand=True)

left_frame = ttk.Frame(paned, width=400)
right_frame = ttk.Frame(paned)
paned.add(left_frame, weight=1)
paned.add(right_frame, weight=3)

listbox = tk.Listbox(left_frame, yscrollcommand=scrollbar.set, font=("Arial", 12))
listbox.pack(fill="both", expand=True)
scrollbar.config(command=listbox.yview)

# Detail pane
detail_text = tk.Text(right_frame, wrap='word', font=("Arial", 12))
detail_text.pack(fill='both', expand=True, padx=6, pady=6)
detail_text.config(state='disabled')

detail_bottom = ttk.Frame(right_frame)
detail_bottom.pack(fill='x')

lbl_group = ttk.Label(detail_bottom, text='Group:')
lbl_group.pack(side='left', padx=4)
group_var = tk.StringVar()
group_entry = ttk.Entry(detail_bottom, textvariable=group_var)
group_entry.pack(side='left', padx=4)

btn_assign = ttk.Button(detail_bottom, text='Assign Group')
btn_assign.pack(side='left', padx=4)

btn_copy = ttk.Button(detail_bottom, text='Copy Text')
btn_copy.pack(side='right', padx=4)

btn_delete_one = ttk.Button(detail_bottom, text='Delete Paste')
btn_delete_one.pack(side='right', padx=4)
btn_ungroup_one = ttk.Button(detail_bottom, text='Ungroup')
btn_ungroup_one.pack(side='right', padx=4)

btn_random = ttk.Button(top_frame, text='Random Paste')
btn_random.pack(side='left', padx=4)
btn_ungrouped = ttk.Button(top_frame, text='Show Ungrouped')
btn_ungrouped.pack(side='left', padx=4)

items_cache = []

def format_preview(item):
    preview = item['text'].replace("\n", " ")[:120]
    ts = datetime.datetime.fromtimestamp(item['time']).strftime('%Y-%m-%d %H:%M:%S')
    cat = item.get('category', 'Pending')
    grp = item.get('group', '')
    locked = item.get('locked', False)
    grp_str = f" ({grp}{' 🔒' if locked else ''})" if grp else ''
    return f"[{cat}]{grp_str} {preview} — {ts}", cat

def update_status():
    if classifier_ready.is_set():
        status_label.config(text="Classifier: Ready" if classifier is not None else "Classifier: Unavailable")
    else:
        status_label.config(text="Classifier: Loading...")

def update_gui():
    update_status()
    listbox.delete(0, tk.END)
    items_cache.clear()
    with db_lock:
        all_items = db.all()
    documents = [dict(it) for it in all_items]
    # apply filter
    f = filter_var.get().lower().strip()
    if f:
        if f == '__ungrouped__':
            documents = [d for d in documents if not d.get('group')]
        else:
            documents = [d for d in documents if f in d.get('text', '').lower() or f in d.get('category', '').lower() or f in d.get('group', '').lower()]
    # sorting
    if sort_var.get() == 'Newest':
        documents.sort(key=lambda x: x.get('time', 0), reverse=True)
    elif sort_var.get() == 'Oldest':
        documents.sort(key=lambda x: x.get('time', 0))
    else:
        documents.sort(key=lambda x: x.get('category', ''))
    for d in documents:
        # prefer stored _doc_id if present
        preview, cat = format_preview(d)
        if d.get('category') == 'Pending':
            preview = preview + ' [Pending]'
        listbox.insert(tk.END, preview)
        items_cache.append(d)

def get_selected_item():
    sel = listbox.curselection()
    if not sel:
        return None
    return items_cache[sel[0]]

def show_detail(event=None):
    it = get_selected_item()
    if not it:
        return
    text = it.get('text', '')
    grp = it.get('group', '')
    group_var.set(grp)
    detail_text.config(state='normal')
    detail_text.delete('1.0', tk.END)
    detail_text.insert(tk.END, text)
    detail_text.config(state='disabled')

def delete_selected():
    it = get_selected_item()
    if not it:
        return
    doc_id = it.get('_doc_id')
    with db_lock:
        if doc_id:
            db.remove(doc_ids=[doc_id])
        else:
            QueryObj = Query()
            db.remove((QueryObj.text == it.get('text')) & (QueryObj.time == it.get('time')))
    update_gui()

def delete_one():
    it = get_selected_item()
    if not it:
        return
    if not messagebox.askyesno('Confirm', 'Delete this paste?'):
        return
    delete_selected()

def clear_all():
    if not messagebox.askyesno('Confirm', 'Delete all saved pastes?'):
        return
    with db_lock:
        db.truncate()
    update_gui()

def copy_text():
    it = get_selected_item()
    if not it:
        return
    try:
        pyperclip.copy(it.get('text', ''))
    except Exception:
        pass

def apply_filter():
    update_gui()

def assign_group():
    it = get_selected_item()
    if not it:
        return
    grp = group_var.get().strip()
    doc_id = it.get('_doc_id')
    with db_lock:
        if doc_id:
            db.update({'group': grp, 'locked': True}, doc_ids=[doc_id])
        else:
            QueryObj = Query()
            db.update({'group': grp, 'locked': True}, (QueryObj.text == it.get('text')) & (QueryObj.time == it.get('time')))
    update_gui()

def ungroup_one():
    it = get_selected_item()
    if not it:
        return
    doc_id = it.get('_doc_id')
    with db_lock:
        if doc_id:
            db.update({'group': '', 'locked': False}, doc_ids=[doc_id])
        else:
            QueryObj = Query()
            db.update({'group': '', 'locked': False}, (QueryObj.text == it.get('text')) & (QueryObj.time == it.get('time')))
    update_gui()

def random_paste():
    import random
    with db_lock:
        all_items = db.all()
    if not all_items:
        return
    it = random.choice(all_items)
    # find index in items_cache to select
    update_gui()
    for idx, d in enumerate(items_cache):
        if d.get('text') == it.get('text') and abs(d.get('time',0)-it.get('time',0))<1:
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(idx)
            listbox.see(idx)
            show_detail()
            return

btn_delete.config(command=delete_selected)
btn_clear.config(command=clear_all)
btn_filter.config(command=apply_filter)
btn_assign.config(command=assign_group)
btn_copy.config(command=copy_text)
btn_delete_one.config(command=delete_one)
btn_random.config(command=random_paste)
btn_ungrouped.config(command=lambda: (filter_var.set('__ungrouped__'), update_gui()))
btn_ungroup_one.config(command=ungroup_one)
listbox.bind('<Double-Button-1>', show_detail)

update_gui()

def auto_group_worker():
    # cluster unlocked items by word-overlap similarity and assign temporary groups
    import collections
    import math
    stopwords = set(['the','and','for','with','that','this','from','your','have','will','are','not','but','you','this','that','they','their','them'])
    # mapping of common keywords to human-friendly topics and weights
    keyword_map = {
        # Formal Sciences
        'logic': 'Formal Sciences', 'mathematics': 'Formal Sciences', 'mathematical': 'Formal Sciences', 'statistics': 'Formal Sciences', 'probability': 'Formal Sciences', 'proof': 'Formal Sciences',
        # Physical Sciences
        'astronomy': 'Physical Sciences', 'chemistry': 'Physical Sciences', 'geology': 'Physical Sciences', 'physics': 'Physical Sciences', 'quantum': 'Physical Sciences', 'particle': 'Physical Sciences', 'relativity': 'Physical Sciences',
        # Life Sciences
        'biology': 'Life Sciences', 'botany': 'Life Sciences', 'ecology': 'Life Sciences', 'neuroscience': 'Life Sciences', 'zoology': 'Life Sciences', 'genetics': 'Life Sciences', 'evolution': 'Life Sciences',
        # Applied Sciences
        'agriculture': 'Applied Sciences', 'architecture': 'Applied Sciences', 'computer': 'Applied Sciences', 'computer science': 'Applied Sciences', 'engineering': 'Applied Sciences', 'medicine': 'Applied Sciences', 'health': 'Applied Sciences', 'technology': 'Applied Sciences',
        # Social Sciences
        'anthropology': 'Social Sciences', 'archaeology': 'Social Sciences', 'criminology': 'Social Sciences', 'economics': 'Social Sciences', 'geography': 'Social Sciences', 'international': 'Social Sciences', 'political': 'Social Sciences', 'psychology': 'Social Sciences', 'sociology': 'Social Sciences',
        # Humanities
        'art': 'Humanities', 'art history': 'Humanities', 'classics': 'Humanities', 'history': 'Humanities', 'linguistics': 'Humanities', 'literature': 'Humanities', 'philosophy': 'Humanities', 'religious': 'Humanities',
        # Professional Disciplines
        'business': 'Professional Disciplines', 'education': 'Professional Disciplines', 'finance': 'Professional Disciplines', 'law': 'Professional Disciplines', 'management': 'Professional Disciplines', 'marketing': 'Professional Disciplines',
        # Arts & Media
        'design': 'Arts & Media', 'film': 'Arts & Media', 'journalism': 'Arts & Media', 'music': 'Arts & Media', 'theater': 'Arts & Media', 'visual': 'Arts & Media',
        # Math
        'add': 'Mathematics', 'subtract': 'Mathematics', 'multiply': 'Mathematics', 'divide': 'Mathematics', 'arithmetic': 'Mathematics', 'algebra': 'Mathematics', 'graph': 'Mathematics', 'geometry': 'Mathematics', 'equation': 'Mathematics', 'matrix': 'Mathematics', 'function': 'Mathematics', 'calculus': 'Mathematics', 'integral': 'Mathematics', 'derivative': 'Mathematics',
        # Programming
        'python': 'Computer Science', 'code': 'Computer Science', 'script': 'Computer Science', 'variable': 'Computer Science', 'loop': 'Computer Science', 'class': 'Computer Science', 'compile': 'Computer Science', 'debug': 'Computer Science',
        # Cooking
        'recipe': 'Cooking', 'cook': 'Cooking', 'bake': 'Cooking', 'ingredient': 'Cooking', 'oven': 'Cooking',
        # School/Study
        'homework': 'School', 'assignment': 'School', 'study': 'School', 'essay': 'School',
        # AI / ML / Research
        'ai': 'AI', 'artificial': 'AI', 'intelligence': 'AI', 'machine': 'AI', 'learning': 'AI', 'ml': 'AI', 'deep': 'AI', 'neural': 'AI', 'network': 'AI', 'neuralnetwork': 'AI', 'transformer': 'AI', 'bert': 'AI', 'gpt': 'AI', 'chatgpt': 'AI', 'model': 'AI', 'inference': 'AI', 'algorithm': 'AI', 'dataset': 'AI', 'training': 'AI', 'nlp': 'AI', 'vision': 'AI', 'reinforcement': 'AI', 'rl': 'AI', 'supervised': 'AI', 'unsupervised': 'AI', 'embedding': 'AI', 'token': 'AI', 'attention': 'AI', 'finetune': 'AI', 'fine-tune': 'AI', 'optimizer': 'AI', 'gradient': 'AI', 'cuda': 'AI', 'torch': 'AI', 'tensorflow': 'AI', 'pytorch': 'AI', 'llm': 'AI', 'language': 'AI',
        # Research / Papers
        'paper': 'Research', 'survey': 'Research', 'experiment': 'Research', 'results': 'Research', 'method': 'Research', 'analysis': 'Research', 'benchmark': 'Research',
    }

    # explicit keyword weights to prioritize some domain words for grouping
    KEYWORD_WEIGHTS = {
        # AI/ML stronger weight
        'ai': 3.0, 'artificial': 2.5, 'intelligence': 2.5, 'machine': 2.0, 'learning': 2.5, 'ml': 2.5, 'deep': 2.0, 'neural': 2.0, 'network': 1.8, 'transformer': 2.5, 'gpt': 3.0, 'chatgpt': 3.0, 'llm': 3.0, 'nlp': 2.0, 'embedding': 2.0,
        # research weight
        'paper': 1.8, 'study': 1.5, 'survey': 1.5, 'experiment': 1.4, 'benchmark': 1.6,
        # math weight
        'algebra': 1.5, 'graph': 1.2, 'equation': 1.5, 'matrix': 1.4,
    }

    def jaccard(a, b):
        if not a or not b:
            return 0.0
        inter = a & b
        uni = a | b
        return len(inter) / len(uni)

    while True:
        try:
            with db_lock:
                all_items = [dict(it) for it in db.all()]
            # consider only unlocked items for auto-grouping
            unlocked = [it for it in all_items if not it.get('locked')]
            n = len(unlocked)
            if n < 2:
                time.sleep(60)
                continue
            # prepare tokenized text for fallback keyword grouping in either mode
            tokens = []
            for it in unlocked:
                words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", it.get('text',''))]
                words = [w for w in words if w not in stopwords]
                tokens.append(set(words))

            # try semantic embeddings clustering if embedder available
            use_embeddings = embed_ready.is_set() and embedder is not None
            vectors = []
            if use_embeddings:
                for it in unlocked:
                    txt = it.get('text','')
                    try:
                        out = embedder(txt)
                        # pipeline feature-extraction returns nested lists; average pooling
                        vec = None
                        if isinstance(out, list):
                            # out may be (1, seq_len, dim) or (seq_len, dim)
                            arr = out[0] if len(out) and isinstance(out[0], list) and isinstance(out[0][0], list) else out
                            # flatten: arr is list of token vectors
                            s = [0.0] * len(arr[0])
                            for token in arr:
                                for k, v in enumerate(token):
                                    s[k] += v
                            vec = [v / len(arr) for v in s]
                        if vec is None:
                            vec = [0.0]
                    except Exception:
                        vec = [0.0]
                    vectors.append(vec)
                # compute similarity via cosine
                parent = list(range(n))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                def cosine(u, v):
                    du = sum(x*x for x in u)
                    dv = sum(x*x for x in v)
                    if du == 0 or dv == 0:
                        return 0.0
                    num = sum(x*y for x,y in zip(u,v))
                    return num / (math.sqrt(du) * math.sqrt(dv))
                THRESH = 0.65
                for i in range(n):
                    for j in range(i+1, n):
                        sim = cosine(vectors[i], vectors[j])
                        if sim >= THRESH:
                            union(i, j)
                clusters = collections.defaultdict(list)
                for i in range(n):
                    clusters[find(i)].append(i)
            else:
                # union-find for clusters by keyword overlap
                parent = list(range(n))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                # threshold for similarity
                THRESH = 0.25
                for i in range(n):
                    for j in range(i+1, n):
                        sim = jaccard(tokens[i], tokens[j])
                        if sim >= THRESH:
                            union(i, j)
                clusters = collections.defaultdict(list)
                for i in range(n):
                    clusters[find(i)].append(i)

            # prepare group assignments
            assignments = []  # list of (doc_id, group_name)
            for root_idx, idxs in clusters.items():
                if len(idxs) < 2:
                    continue
                # collect word frequencies
                wc = collections.Counter()
                for idx in idxs:
                    wc.update(tokens[idx])
                if not wc:
                    continue
                # score candidate groups using keyword_map + weights
                group_scores = collections.Counter()
                for word, count in wc.items():
                    mapped = keyword_map.get(word)
                    weight = KEYWORD_WEIGHTS.get(word, 1.0)
                    if mapped:
                        group_scores[mapped] += count * weight
                if group_scores:
                    grp_name, _ = group_scores.most_common(1)[0]
                else:
                    cluster_text = " ".join(unlocked[idx].get('text','') for idx in idxs)
                    grp_name = get_cluster_topic(cluster_text)
                    if not grp_name:
                        most_common_word, _ = wc.most_common(1)[0]
                        grp_name = keyword_map.get(most_common_word, f"Group: {most_common_word}")
                for idx in idxs:
                    it = unlocked[idx]
                    doc_id = it.get('_doc_id')
                    assignments.append((doc_id, grp_name, it.get('text'), it.get('time')))

            # apply assignments (only to unlocked items) and clear groups no longer in any assignment
            assigned_doc_ids = set(d for d,_,_,_ in assignments if d)
            with db_lock:
                # set groups for assigned doc_ids
                for doc_id, grp_name, text_val, time_val in assignments:
                    if doc_id:
                        db.update({'group': grp_name}, doc_ids=[doc_id])
                    else:
                        QueryObj = Query()
                        db.update({'group': grp_name}, (QueryObj.text == text_val) & (QueryObj.time == time_val))
                # clear group for unlocked items not assigned
                for it in unlocked:
                    doc_id = it.get('_doc_id')
                    if doc_id and doc_id not in assigned_doc_ids and it.get('group'):
                        db.update({'group': ''}, doc_ids=[doc_id])
            try:
                root.after(0, update_gui)
            except Exception:
                pass
        except Exception:
            traceback.print_exc()
        time.sleep(60)

import re
threading.Thread(target=auto_group_worker, daemon=True).start()

root.mainloop()
