import streamlit as st
import time
import torch
import pickle
import re
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# --- PAGE CONFIG ---
# We natively remove 'About' and 'Help' items here to reduce menu bloat
st.set_page_config(
    page_title="BIS Compliance Copilot",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# --- CUSTOM CSS FOR PREMIUM LOOK & FORM STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&family=Montserrat:wght@600;700&display=swap');
    
    /* Apply Inter font globally */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Prevent 'Inter' from overwriting Material Icons */
    .material-symbols-rounded, 
    .material-symbols-outlined, [class*="material-symbols"],[data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }

    /* Brighter premium blue for titles */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #004B87 !important; 
    }
    .main {
        background-color: #f8f9fa;
    }
    
    /* Button Styling (Applies to regular buttons AND form submit buttons) */
    .stButton>button, [data-testid="stFormSubmitButton"] > button {
        background-color: #004B87;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #003366;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: white;
    }
    
    /* Result Cards */
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 5px solid #004B87;
    }
    
    /* Premium Font for the Standard IDs */
    .standard-title {
        font-family: 'Montserrat', sans-serif !important;
        color: #004B87;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    /* Squircle Metric Badge */
    .metric-badge {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Make the Form container invisible so it blends perfectly */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* --- HIDE DEPLOY BUTTON --- */
    .stAppDeployButton {
        display: none !important;
    }
    
    /* --- STREAMLIT 3-DOTS MENU ISOLATION (KEEP ONLY THEME) --- */
    /* This aggressively guts the menu popup, stripping lists, dividers, and footers */
    div[data-baseweb="popover"] ul, 
    div[data-baseweb="popover"] ul[role="menu"],
    div[data-baseweb="popover"] [data-baseweb="divider"],
    div[data-baseweb="popover"] hr,
    div[data-baseweb="popover"] a {
        display: none !important;
    }
    /* Ensure the popover loses its bottom padding since we removed the footer */
    div[data-baseweb="popover"] > div > div {
        padding-bottom: 0.5rem !important;
    }
    
    /* CHATGPT STYLE SIDEBAR HISTORY EXPANDERS */
    [data-testid="stSidebar"][data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        margin-bottom: 0.2rem !important;
    }[data-testid="stSidebar"][data-testid="stExpander"] details {
        border: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.5rem !important;
        transition: background-color 0.2s;
    }
    [data-testid="stSidebar"][data-testid="stExpander"] summary:hover {
        background-color: rgba(0, 75, 135, 0.08) !important;
    }
    
    /* Custom Chevron for Expanders */
    [data-testid="stExpanderToggleIcon"] > * {
        display: none !important;
    }
    [data-testid="stExpanderToggleIcon"]::before {
        content: '\e76c'; 
        font-family: 'Segoe UI Symbol', 'Arial', sans-serif;
        display: inline-block;
        font-size: 1.2em;
        color: #004B87;
        vertical-align: middle;
        transition: transform 0.2s;
    }
    [data-testid="stExpander"] [aria-expanded="true"] [data-testid="stExpanderToggleIcon"]::before {
        transform: rotate(90deg);
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHE MODELS ---
@st.cache_resource(show_spinner=True)
def load_system():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', 
                            activation_fn=torch.nn.Sigmoid(), 
                            device=device)
    with open('data/knowledge_base.pkl', 'rb') as f:
        data = pickle.load(f)
    return embedder, reranker, data['corpus'], data['metadata'], torch.tensor(data['embeddings'], device=device), device

def normalize_standard_id(std_id):
    clean_id = re.sub(r'\s+', ' ', std_id).strip()
    match = re.search(r'IS\s*(\d+)\s*(?:\(\s*Part\s*(\d+)\s*\))?\s*:\s*(\d{4})', clean_id)
    if match:
        part = match.group(2)
        if part:
            return f"IS {match.group(1)} (Part {part}): {match.group(3)}"
        else:
            return f"IS {match.group(1)}: {match.group(3)}"
    return clean_id

try:
    embedder, reranker, corpus, metadata, corpus_embeddings, device = load_system()
except FileNotFoundError:
    st.error("Knowledge base not found. Please run `src/ingest.py` first to process the PDF.")
    st.stop()

# --- INITIALIZE CHAT HISTORY ---
if "history" not in st.session_state:
    st.session_state.history =[]

# --- SIDEBAR ---
with st.sidebar:
    import os
    if os.path.exists("bis_logo.png"):
        st.image("bis_logo.png", width=120)
    else:
        st.info("💡 Place the official logo as 'bis_logo.png' in the root directory.")
    
    st.title("MSE Compliance")
    st.markdown("**Transforming BIS discovery for Micro & Small Enterprises.**")
    st.caption("Pipeline: Pure Dense Search + Cross-Encoder Reranking")
    st.caption("Latency: < 1.0 second")
    
    st.divider()
    
    # CHATGPT-STYLE HISTORY RENDERED IN SIDEBAR
    st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #555; margin-bottom: 0.5rem;'>Previous Analyses</div>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.caption("Your search history will appear here.")
    else:
        for i, run in enumerate(st.session_state.history):
            snippet = run['query'][:28] + "..." if len(run['query']) > 28 else run['query']
            with st.expander(f"💬 {snippet}"):
                st.caption(f"**Latency:** {run['latency']:.2f}s")
                st.markdown(f"**Query:** {run['query']}")
                st.markdown("**Top Results:**")
                for std_id, score in run['results']:
                    st.markdown(f"- {std_id} *({score:.2f})*")

# --- MAIN APP ---
st.title("BIS Standard Discovery Copilot")
st.markdown("<p style='font-size:1.1rem; color:#555;'>Describe your product or manufacturing intent, and our AI will instantly recommend the exact Bureau of Indian Standards (IS) compliance code(s) you need.</p>", unsafe_allow_html=True)

# Using st.form automatically maps Ctrl+Enter (or Cmd+Enter) inside the text_area to trigger the form submit button
with st.form("search_form"):
    query = st.text_area(
        "Please enter your query (e.g., product description, manufacturing process, etc.):", 
        placeholder="e.g., We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement...", 
        height=100
    )
    submitted = st.form_submit_button("Analyze Compliance Requirements")

if submitted:
    if not query.strip():
        st.warning("Please describe your product first.")
    else:
        st.divider()
        start_time = time.time()
        
        with st.spinner("Searching millions of reference parameters..."):
            # Dense Retrieval
            query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)
            dense_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=20)[0]
            top_indices = [hit['corpus_id'] for hit in dense_hits]
            
            # Reranking
            cross_inp = [[query, corpus[idx]] for idx in top_indices]
            cross_scores = reranker.predict(cross_inp)
            ranked_pairs = sorted(zip(top_indices, cross_scores), key=lambda x: x[1], reverse=True)
            
            # Aggregate & Dedup
            standard_scores = {}
            for rank, (idx, score) in enumerate(ranked_pairs):
                std_id = normalize_standard_id(metadata[idx]["Standard_ID"])
                if std_id not in standard_scores:
                    standard_scores[std_id] = float(score)
                else:
                    standard_scores[std_id] += float(score) * 0.1
                    
            sorted_standards = sorted(standard_scores.items(), key=lambda x: x[1], reverse=True)
            top_results = sorted_standards[:5]
            
            latency = time.time() - start_time
            
            # Save query and results to history
            st.session_state.history.insert(0, {
                "query": query,
                "latency": latency,
                "results": top_results
            })
            
            # Re-running the UI isn't necessary here since `st.form` naturally reruns the script upon submission!

# --- DISPLAY LATEST RESULT IN MAIN AREA ---
if st.session_state.history:
    latest_run = st.session_state.history[0]
    
    st.success(f"Analysis complete in {latest_run['latency']:.2f} seconds.")
    st.markdown("### Recommended Standards")
    
    for std_id, score in latest_run['results']:
        st.markdown(f"""
        <div class="result-card">
            <div class="standard-title">Standard {std_id}</div>
            <div class="metric-badge">Relevance Score: {score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)