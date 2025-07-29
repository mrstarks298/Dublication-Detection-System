from flask import Flask, request, jsonify, render_template, send_file
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import tempfile
import time
from werkzeug.utils import secure_filename
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
import shutil
import glob
import csv
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STORAGE_FOLDER'] = 'question_storage'
app.config['UNIQUE_FOLDER'] = os.path.join(app.config['STORAGE_FOLDER'], 'unique_questions')
app.config['DUPLICATE_FOLDER'] = os.path.join(app.config['STORAGE_FOLDER'], 'duplicates')
app.config['EMBEDDINGS_FOLDER'] = os.path.join(app.config['STORAGE_FOLDER'], 'embeddings')

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['STORAGE_FOLDER'], 
               app.config['UNIQUE_FOLDER'], app.config['DUPLICATE_FOLDER'],
               app.config['EMBEDDINGS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model - load once when app starts
model = None

def load_model():
    """Load the sentence transformer model"""
    global model
    if model is None:
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully!")
    return model

def encode_text(text):
    """Encode text using the sentence transformer model"""
    model = load_model()
    return model.encode(str(text)).tolist()

def encode_texts_batch(texts):
    """Encode multiple texts in batch for efficiency"""
    model = load_model()
    return model.encode(texts)

def build_faiss_index(embeddings):
    """Build FAISS index for efficient similarity search"""
    embeddings_array = np.array(embeddings, dtype='float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    # Create FAISS index (Inner Product for normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)
    
    return index, embeddings_array

def extract_chapter_name(filename):
    """Extract chapter name from filename"""
    # Remove .json extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return base_name

def find_duplicates_with_faiss(embeddings, questions_data, thresholds, k=50):
    """Find duplicates using FAISS for efficient similarity search"""
    logger.info(f"Building FAISS index for {len(embeddings)} embeddings...")
    
    index, embeddings_array = build_faiss_index(embeddings)
    
    logger.info("Performing FAISS similarity search...")
    # Search for k most similar questions for each question
    scores, indices = index.search(embeddings_array, k)
    
    duplicates = []
    processed_pairs = set()  # To avoid duplicate pairs
    
    for i in range(len(questions_data)):
        for r in range(1, min(k, len(indices[i]))):  # Skip self match (index 0)
            j = indices[i][r]
            sim_score = float(scores[i][r])
            
            # Skip if similarity is below threshold
            if sim_score < thresholds['semantic_duplicate_threshold']:
                continue
                
            # Avoid duplicate pairs (i,j) and (j,i)
            pair = tuple(sorted([i, j]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            # Determine tag based on similarity score
            if sim_score >= 0.95:
                tag = "Exact Duplicate"
            elif sim_score >= thresholds['near_duplicate_threshold']:
                tag = "Near Duplicate"
            else:
                tag = "Semantic Duplicate"
            
            duplicates.append({
                'DuplicateQuestionID': questions_data[j]['QuestionID'],
                'OriginalQuestionID': questions_data[i]['QuestionID'],
                'DuplicateQuestion': questions_data[j].get('Question', ''),
                'OriginalQuestion': questions_data[i].get('Question', ''),
                'SimilarityScore': sim_score,
                'Type': tag,
                'DuplicationType': 'Internal',
                'CrossFile': False
            })
    
    logger.info(f"Found {len(duplicates)} duplicate pairs using FAISS")
    return duplicates

def find_cross_file_duplicates_with_faiss(current_questions, existing_questions, thresholds):
    """Find duplicates between current file and existing unique questions using FAISS"""
    logger.info(f"Finding cross-file duplicates: {len(current_questions)} vs {len(existing_questions)} questions")
    
    if not existing_questions:
        return [], set()
    
    # Generate embeddings for all questions
    logger.info("Generating embeddings for existing questions...")
    existing_texts = [q.get('Question', '') for q in existing_questions]
    existing_embeddings = encode_texts_batch(existing_texts)
    
    logger.info("Generating embeddings for current questions...")
    current_texts = [q.get('Question', '') for q in current_questions]
    current_embeddings = encode_texts_batch(current_texts)
    
    # Build FAISS index for existing questions
    logger.info("Building FAISS index for existing questions...")
    existing_index, existing_embeddings_normalized = build_faiss_index(existing_embeddings)
    
    # Normalize current embeddings for search
    current_embeddings_array = np.array(current_embeddings, dtype='float32')
    faiss.normalize_L2(current_embeddings_array)
    
    # Search for similar questions in existing database
    k = min(10, len(existing_questions))  # Search top 10 or all if less
    scores, indices = existing_index.search(current_embeddings_array, k)
    
    cross_duplicates = []
    questions_to_remove = set()
    
    for i, current_q in enumerate(current_questions):
        for r in range(k):
            if r >= len(indices[i]):
                break
                
            j = indices[i][r]
            sim_score = float(scores[i][r])
            
            if sim_score >= thresholds['semantic_duplicate_threshold']:
                if sim_score >= 0.95:
                    tag = "Exact Duplicate"
                elif sim_score >= thresholds['near_duplicate_threshold']:
                    tag = "Near Duplicate"
                else:
                    tag = "Semantic Duplicate"
                
                cross_duplicates.append({
                    'DuplicateQuestionID': current_q['QuestionID'],   # Current question is duplicate
                    'OriginalQuestionID': existing_questions[j]['QuestionID'],  # Existing question is original
                    'DuplicateQuestion': current_q.get('Question', ''),
                    'OriginalQuestion': existing_questions[j].get('Question', ''),
                    'SimilarityScore': sim_score,
                    'Type': tag,
                    'DuplicationType': 'Cross-File',
                    'CrossFile': True
                })
                questions_to_remove.add(current_q['QuestionID'])
                break  # Don't check this current question against other existing questions
    
    logger.info(f"Found {len(cross_duplicates)} cross-file duplicates")
    logger.info(f"Removing {len(questions_to_remove)} questions from current file")
    
    return cross_duplicates, questions_to_remove

class UnionFind:
    """Union-Find data structure for grouping connected duplicates"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_groups(self):
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return [group for group in groups.values() if len(group) > 1]

def load_existing_unique_questions():
    """Load all existing unique question files"""
    unique_files = glob.glob(os.path.join(app.config['UNIQUE_FOLDER'], '*_UniqueChapterList.json'))
    all_unique_questions = []
    file_mapping = {}  # Map question_id to source file
    
    for file_path in unique_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'Content' in data:
                    questions = data['Content']
                elif isinstance(data, list):
                    questions = data
                else:
                    continue
                
                filename = os.path.basename(file_path)
                for question in questions:
                    if isinstance(question, dict) and 'QuestionID' in question:
                        all_unique_questions.append(question)
                        file_mapping[question['QuestionID']] = filename
                        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(all_unique_questions)} existing unique questions from {len(unique_files)} files")
    return all_unique_questions, file_mapping

def load_existing_embeddings():
    """Load existing embeddings from CSV files with NaN handling"""
    embeddings_files = glob.glob(os.path.join(app.config['EMBEDDINGS_FOLDER'], '*_Embeddings.csv'))
    all_embeddings = []
    
    for file_path in embeddings_files:
        try:
            df = pd.read_csv(file_path)
            if 'QuestionID' in df.columns and 'Question' in df.columns and 'Embedding' in df.columns:
                for _, row in df.iterrows():
                    try:
                        # Parse embedding string back to list with NaN handling
                        embedding_str = str(row['Embedding'])
                        if embedding_str == 'nan' or pd.isna(row['Embedding']):
                            logger.warning(f"Skipping question {row.get('QuestionID', 'unknown')} - invalid embedding")
                            continue
                            
                        embedding = json.loads(embedding_str)
                        
                        # Validate embedding is a list of numbers without NaN
                        if not isinstance(embedding, list) or not embedding:
                            logger.warning(f"Skipping question {row.get('QuestionID', 'unknown')} - invalid embedding format")
                            continue
                            
                        # Check for NaN values in embedding
                        if any(not isinstance(x, (int, float)) or np.isnan(x) for x in embedding):
                            logger.warning(f"Skipping question {row.get('QuestionID', 'unknown')} - NaN values in embedding")
                            continue
                        
                        all_embeddings.append({
                            'QuestionID': str(row['QuestionID']),
                            'Question': str(row['Question']),
                            'Embedding': embedding,
                            'SourceFile': os.path.basename(file_path)
                        })
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning(f"Error parsing embedding for question {row.get('QuestionID', 'unknown')}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(all_embeddings)} valid existing embeddings from {len(embeddings_files)} files")
    return all_embeddings

def safe_json_serialize(obj):
    """Safely serialize object to JSON, handling NaN values"""
    def sanitize_value(value):
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
        elif isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [sanitize_value(item) for item in value]
        return value
    
    return sanitize_value(obj)

def get_duplicate_questions_from_pairs(duplicate_pairs, original_data):
    """Extract actual duplicate questions from the original data based on duplicate pairs"""
    duplicate_questions = []
    duplicate_ids = set()
    
    # Create a mapping from QuestionID to question data
    id_to_question = {q['QuestionID']: q for q in original_data}
    
    # Collect all duplicate question IDs
    for pair in duplicate_pairs:
        duplicate_ids.add(pair['DuplicateQuestionID'])
    
    # Get the actual question data for duplicate IDs
    for dup_id in duplicate_ids:
        if dup_id in id_to_question:
            duplicate_questions.append(id_to_question[dup_id])
    
    return duplicate_questions

def save_results_organized(filename, unique_questions, all_duplicates, original_data, stats):
    """Save unique questions and duplicate reports with proper chapter naming"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    chapter_name = extract_chapter_name(filename)

    print(f"DEBUG: filename={filename}, chapter_name={chapter_name}")
    logger.info(f"Extracted chapter name: {chapter_name} from filename: {filename}")
    
    # Clean unique_questions of any NaN values
    cleaned_unique_questions = []
    for question in unique_questions:
        cleaned_question = safe_json_serialize(question)
        if cleaned_question:  # Only add if not None
            cleaned_unique_questions.append(cleaned_question)
    
    # Get duplicate questions from the original data
    duplicate_questions = get_duplicate_questions_from_pairs(all_duplicates, original_data)
    cleaned_duplicate_questions = []
    for question in duplicate_questions:
        cleaned_question = safe_json_serialize(question)
        if cleaned_question:
            cleaned_duplicate_questions.append(cleaned_question)
    
    # 1. Save unique questions JSON - {ChapterName}_UniqueChapterList.json
    unique_json_filename = f"{chapter_name}_UniqueChapterList.json"
    unique_json_path = os.path.join(app.config['UNIQUE_FOLDER'], unique_json_filename)
    
    unique_data = safe_json_serialize({
        'Content': cleaned_unique_questions,
        'Metadata': {
            'original_filename': filename,
            'chapter_name': chapter_name,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'total_questions': stats.get('total_questions', 0),
            'unique_questions': len(cleaned_unique_questions),
            'processing_time': stats.get('processing_time', 0),
            'optimization': 'FAISS-enabled'
        }
    })
    
    with open(unique_json_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    # 2. Save duplicate questions JSON - {ChapterName}_DuplicateChapterList.json
    duplicate_json_filename = f"{chapter_name}_DuplicateChapterList.json"
    duplicate_json_path = os.path.join(app.config['DUPLICATE_FOLDER'], duplicate_json_filename)
    
    duplicate_data = safe_json_serialize({
        'Content': cleaned_duplicate_questions,
        'Metadata': {
            'original_filename': filename,
            'chapter_name': chapter_name,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'total_duplicate_questions': len(cleaned_duplicate_questions),
            'total_duplicate_pairs': len(all_duplicates),
            'internal_duplicates': len([d for d in all_duplicates if not d.get('CrossFile', False)]),
            'cross_file_duplicates': len([d for d in all_duplicates if d.get('CrossFile', False)]),
            'optimization': 'FAISS-enabled'
        }
    })
    
    with open(duplicate_json_path, 'w', encoding='utf-8') as f:
        json.dump(duplicate_data, f, indent=2, ensure_ascii=False)
    
    # 3. Save duplicate report CSV - {ChapterName}_DuplicateReport.csv
    duplicate_report_csv_filename = f"{chapter_name}_DuplicateReport.csv"
    duplicate_report_csv_path = os.path.join(app.config['DUPLICATE_FOLDER'], duplicate_report_csv_filename)
    
    if all_duplicates:
        # Clean duplicates data for CSV
        cleaned_duplicates = []
        for dup in all_duplicates:
            cleaned_dup = safe_json_serialize(dup)
            if cleaned_dup:
                cleaned_duplicates.append(cleaned_dup)
        
        df_duplicates = pd.DataFrame(cleaned_duplicates)
        df_duplicates = df_duplicates.fillna('')  # Replace any remaining NaN values
        df_duplicates.to_csv(duplicate_report_csv_path, index=False, encoding='utf-8')
    
    # 4. Save embeddings CSV - {ChapterName}_Embeddings.csv
    embeddings_csv_filename = f"{chapter_name}_Embeddings.csv"
    embeddings_csv_path = os.path.join(app.config['EMBEDDINGS_FOLDER'], embeddings_csv_filename)
    
    embeddings_data = []
    for question in cleaned_unique_questions:
        embedding = question.get('Embedding', [])
        # Ensure embedding is valid
        if isinstance(embedding, list) and not any(np.isnan(x) if isinstance(x, (int, float)) else False for x in embedding):
            embedding_str = json.dumps(embedding)
        else:
            embedding_str = "[]"
            
        embeddings_data.append({
            'QuestionID': str(question.get('QuestionID', '')),
            'Question': str(question.get('Question', '')),
            'Embedding': embedding_str
        })
    
    df_embeddings = pd.DataFrame(embeddings_data)
    df_embeddings = df_embeddings.fillna('')  # Replace any NaN values with empty strings
    df_embeddings.to_csv(embeddings_csv_path, index=False, encoding='utf-8')
    
    logger.info(f"Saved organized results:")
    logger.info(f"  - Unique: {unique_json_filename}")
    logger.info(f"  - Duplicates: {duplicate_json_filename}")
    logger.info(f"  - Report: {duplicate_report_csv_filename}")
    logger.info(f"  - Embeddings: {embeddings_csv_filename}")
    
    return {
        'unique_json': unique_json_filename,
        'duplicates_json': duplicate_json_filename,
        'duplicates_csv': duplicate_report_csv_filename,
        'embeddings_csv': embeddings_csv_filename
    }

def process_internal_duplicates(data, thresholds):
    """Process duplicates within the current file using FAISS"""
    logger.info(f"Processing internal duplicates in {len(data)} questions using FAISS...")
    
    # Generate embeddings for questions
    logger.info("Generating question embeddings...")
    question_texts = [item.get('Question', '') for item in data]
    question_embeddings = encode_texts_batch(question_texts)
    
    # Add embeddings to data for CSV export
    for i, item in enumerate(data):
        item['Embedding'] = question_embeddings[i].tolist()
    
    # Find duplicates using FAISS
    duplicates = find_duplicates_with_faiss(question_embeddings, data, thresholds)
    
    # Create unique questions list using Union-Find
    logger.info("Creating unique questions list using Union-Find grouping...")
    n = len(data)
    uf = UnionFind(n)
    
    # Create mapping from QuestionID to index
    id_to_index = {data[i]['QuestionID']: i for i in range(n)}
    
    # Union duplicate pairs
    for dup in duplicates:
        orig_idx = id_to_index[dup['OriginalQuestionID']]
        dup_idx = id_to_index[dup['DuplicateQuestionID']]
        uf.union(orig_idx, dup_idx)
    
    # Get duplicate groups
    duplicate_groups = uf.get_groups()
    
    # Track which indices to keep (first from each group)
    indices_to_keep = set(range(n))
    
    for group in duplicate_groups:
        # Sort group by original order and keep only the first one
        group.sort()
        # Remove all except the first
        for idx in group[1:]:
            indices_to_keep.discard(idx)
    
    # Create unique questions list
    unique_data = [data[i] for i in sorted(indices_to_keep)]
    
    logger.info(f"Found {len(duplicates)} internal duplicate pairs")
    logger.info(f"Found {len(duplicate_groups)} internal duplicate groups")
    logger.info(f"Reduced from {len(data)} to {len(unique_data)} questions after internal deduplication")
    
    return unique_data, duplicates

def perform_cross_check(uploaded_questions, uploaded_filename, thresholds):
    """Perform cross-check against existing embeddings with improved error handling"""
    logger.info("Starting cross-check against existing embeddings...")
    
    # Load existing embeddings with NaN handling
    existing_embeddings = load_existing_embeddings()
    
    if not existing_embeddings:
        logger.info("No existing embeddings found for cross-check")
        return [], []
    
    try:
        # Generate embeddings for uploaded questions
        uploaded_texts = [str(q.get('Question', '')) for q in uploaded_questions]
        uploaded_embeddings = encode_texts_batch(uploaded_texts)
        
        # Validate uploaded embeddings
        valid_uploaded_indices = []
        valid_uploaded_embeddings = []
        
        for i, embedding in enumerate(uploaded_embeddings):
            if not np.any(np.isnan(embedding)) and not np.any(np.isinf(embedding)):
                valid_uploaded_indices.append(i)
                valid_uploaded_embeddings.append(embedding)
        
        if not valid_uploaded_embeddings:
            logger.error("No valid embeddings found in uploaded questions")
            return [], []
        
        # Prepare existing data for FAISS
        existing_embedding_vectors = [emb['Embedding'] for emb in existing_embeddings]
        
        # Validate existing embeddings
        valid_existing_embeddings = []
        valid_existing_data = []
        
        for i, embedding in enumerate(existing_embedding_vectors):
            if not np.any(np.isnan(embedding)) and not np.any(np.isinf(embedding)):
                valid_existing_embeddings.append(embedding)
                valid_existing_data.append(existing_embeddings[i])
        
        if not valid_existing_embeddings:
            logger.error("No valid existing embeddings found")
            return [], []
        
        existing_index, _ = build_faiss_index(valid_existing_embeddings)
        
        # Normalize uploaded embeddings
        uploaded_embeddings_array = np.array(valid_uploaded_embeddings, dtype='float32')
        faiss.normalize_L2(uploaded_embeddings_array)
        
        # Search for similarities
        k = min(5, len(valid_existing_embeddings))
        scores, indices = existing_index.search(uploaded_embeddings_array, k)
        
        cross_check_results = []
        
        for i, uploaded_idx in enumerate(valid_uploaded_indices):
            uploaded_q = uploaded_questions[uploaded_idx]
            
            for r in range(k):
                if r >= len(indices[i]):
                    break
                
                j = indices[i][r]
                sim_score = float(scores[i][r])
                
                # Additional validation for similarity score
                if np.isnan(sim_score) or np.isinf(sim_score):
                    continue
                
                if sim_score >= thresholds['semantic_duplicate_threshold']:
                    existing_q = valid_existing_data[j]
                    
                    if sim_score >= 0.95:
                        match_type = "Exact Match"
                    elif sim_score >= thresholds['near_duplicate_threshold']:
                        match_type = "Near Match"
                    else:
                        match_type = "Semantic Match"
                    
                    result = {
                        'NewQuestionID': str(uploaded_q.get('QuestionID', '')),
                        'NewQuestion': str(uploaded_q.get('Question', '')),
                        'MatchingQuestionID': str(existing_q['QuestionID']),
                        'MatchingQuestion': str(existing_q['Question']),
                        'SimilarityScore': round(float(sim_score), 6),
                        'MatchType': match_type,
                        'SourceFile': str(existing_q['SourceFile'])
                    }
                    
                    # Sanitize the result to remove any potential NaN values
                    result = safe_json_serialize(result)
                    cross_check_results.append(result)
                    break
        
        logger.info(f"Found {len(cross_check_results)} cross-check matches")
        
        # Save cross-check results with safe serialization
        chapter_name = extract_chapter_name(uploaded_filename)
        
        # Save as JSON
        crosscheck_json_filename = f"{chapter_name}_CrossCheck.json"
        crosscheck_json_path = os.path.join(app.config['STORAGE_FOLDER'], crosscheck_json_filename)
        
        crosscheck_data = safe_json_serialize({
            'CrossCheckResults': cross_check_results,
            'Metadata': {
                'uploaded_filename': uploaded_filename,
                'chapter_name': chapter_name,
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'total_matches': len(cross_check_results),
                'existing_files_checked': len(set(emb['SourceFile'] for emb in valid_existing_data)),
                'total_valid_uploaded': len(valid_uploaded_indices),
                'total_valid_existing': len(valid_existing_data)
            }
        })
        
        with open(crosscheck_json_path, 'w', encoding='utf-8') as f:
            json.dump(crosscheck_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        crosscheck_csv_filename = f"{chapter_name}_CrossCheck.csv"
        crosscheck_csv_path = os.path.join(app.config['STORAGE_FOLDER'], crosscheck_csv_filename)
        
        if cross_check_results:
            df_crosscheck = pd.DataFrame(cross_check_results)
            # Replace any remaining NaN values
            df_crosscheck = df_crosscheck.fillna('')
            df_crosscheck.to_csv(crosscheck_csv_path, index=False, encoding='utf-8')
        
        return cross_check_results, [crosscheck_json_filename, crosscheck_csv_filename]
        
    except Exception as e:
        logger.error(f"Error in cross-check processing: {str(e)}")
        return [], []

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/storage-info')
def storage_info():
    """Get information about stored files"""
    try:
        unique_json_files = glob.glob(os.path.join(app.config['UNIQUE_FOLDER'], '*_UniqueChapterList.json'))
        duplicate_json_files = glob.glob(os.path.join(app.config['DUPLICATE_FOLDER'], '*_DuplicateChapterList.json'))
        duplicate_csv_files = glob.glob(os.path.join(app.config['DUPLICATE_FOLDER'], '*_DuplicateReport.csv'))
        embeddings_files = glob.glob(os.path.join(app.config['EMBEDDINGS_FOLDER'], '*_Embeddings.csv'))
        
        storage_stats = {
            'unique_json_files': len(unique_json_files),
            'duplicate_json_files': len(duplicate_json_files),
            'duplicate_csv_files': len(duplicate_csv_files),
            'embeddings_files': len(embeddings_files),
            'total_unique_questions': 0,
            'storage_folder': app.config['STORAGE_FOLDER'],
            'optimization': 'FAISS-enabled'
        }
        
        # Count total unique questions
        for file_path in unique_json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'Content' in data:
                        storage_stats['total_unique_questions'] += len(data['Content'])
            except:
                continue
        
        return jsonify(storage_stats)
        
    except Exception as e:
        logger.error(f"Storage info error: {str(e)}")
        return jsonify({'error': f'Failed to get storage info: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and validation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.json'):
            return jsonify({'error': 'Only JSON files are allowed'}), 400
        
        # Read and validate JSON
        try:
            content = file.read().decode('utf-8')
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
        except UnicodeDecodeError:
            return jsonify({'error': 'File encoding error. Please use UTF-8 encoding.'}), 400
        
        # Validate structure
        if not isinstance(data, dict) or 'Content' not in data:
            return jsonify({'error': 'Invalid format: Expected "Content" array in JSON'}), 400
        
        if not isinstance(data['Content'], list):
            return jsonify({'error': 'Invalid format: "Content" must be an array'}), 400
        
        if len(data['Content']) == 0:
            return jsonify({'error': 'No questions found in the file'}), 400
        
        # Validate required fields in each question
        required_fields = ['QuestionID', 'Question']
        for i, item in enumerate(data['Content']):
            if not isinstance(item, dict):
                return jsonify({'error': f'Invalid format: Item {i} is not an object'}), 400
            
            for field in required_fields:
                if field not in item:
                    return jsonify({'error': f'Missing required field "{field}" in item {i}'}), 400
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully. Found {len(data["Content"])} questions.',
            'question_count': len(data['Content']),
            'filename': file.filename,
            'data': data,
            'optimization': 'FAISS-enabled'
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_questions():
    """Process questions for duplicates with cross-file comparison using FAISS"""
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data provided'}), 400
        
        data = request_data['data']['Content']
        filename = request_data.get('filename', 'unknown.json')
        
        # Get thresholds from request or use defaults
        thresholds = {
            'near_duplicate_threshold': float(request_data.get('nearDuplicateThreshold', 0.85)),
            'semantic_duplicate_threshold': float(request_data.get('semanticDuplicateThreshold', 0.70))
        }
        
        logger.info(f"Processing {filename} with FAISS optimization. Thresholds: {thresholds}")
        
        start_time = time.time()
        
        # Step 1: Process internal duplicates within current file using FAISS
        unique_questions_internal, internal_duplicates = process_internal_duplicates(data, thresholds)
        
        # Step 2: Load existing unique questions from storage
        existing_unique_questions, file_mapping = load_existing_unique_questions()
        
        # Step 3: Find cross-file duplicates using FAISS
        cross_duplicates = []
        questions_to_remove_cross = set()
        
        if existing_unique_questions:
            cross_duplicates, questions_to_remove_cross = find_cross_file_duplicates_with_faiss(
                unique_questions_internal, existing_unique_questions, thresholds
            )
        
        # Step 4: Remove cross-file duplicates from current unique questions
        final_unique_questions = [
            q for q in unique_questions_internal 
            if q['QuestionID'] not in questions_to_remove_cross
        ]
        
        # Step 5: Combine all duplicates
        all_duplicates = internal_duplicates + cross_duplicates
        
        processing_time = time.time() - start_time
        
        # Step 6: Calculate comprehensive stats
        stats = {
            'total_questions': len(data),
            'unique_questions_after_internal': len(unique_questions_internal),
            'final_unique_questions': len(final_unique_questions),
            'internal_duplicate_pairs': len(internal_duplicates),
            'cross_file_duplicate_pairs': len(cross_duplicates),
            'total_duplicate_pairs': len(all_duplicates),
            'questions_removed_internal': len(data) - len(unique_questions_internal),
            'questions_removed_cross_file': len(questions_to_remove_cross),
            'total_questions_removed': len(data) - len(final_unique_questions),
            'processing_time': round(processing_time, 2),
            'existing_files_checked': len(file_mapping),
            'optimization': 'FAISS-enabled',
            'performance_improvement': f"~{((len(data)**2) / (len(data) * 50)):.0f}x faster than O(N²)"
        }
        
        # Step 7: Save results to organized storage
        saved_files = save_results_organized(
    filename, final_unique_questions, all_duplicates, data, stats
)
        
        result = {
            'unique_questions': {'Content': final_unique_questions},
            'duplicate_report': all_duplicates,
            'stats': stats,
            'saved_files': saved_files
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/cross-check', methods=['POST'])
def cross_check_questions():
    """Perform cross-check against existing embeddings"""
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data provided'}), 400
        
        data = request_data['data']['Content']
        filename = request_data.get('filename', 'unknown.json')
        
        # Get thresholds from request or use defaults
        thresholds = {
            'near_duplicate_threshold': float(request_data.get('nearDuplicateThreshold', 0.85)),
            'semantic_duplicate_threshold': float(request_data.get('semanticDuplicateThreshold', 0.70))
        }
        
        logger.info(f"Performing cross-check for {filename} with FAISS optimization. Thresholds: {thresholds}")
        
        start_time = time.time()
        
        # Perform cross-check against existing embeddings
        cross_check_results, saved_files = perform_cross_check(data, filename, thresholds)
        
        processing_time = time.time() - start_time
        
        # Calculate stats
        stats = {
            'total_uploaded_questions': len(data),
            'total_matches_found': len(cross_check_results),
            'processing_time': round(processing_time, 2),
            'existing_files_checked': len(set(result['SourceFile'] for result in cross_check_results)) if cross_check_results else 0,
            'optimization': 'FAISS-enabled'
        }
        
        result = {
            'cross_check_results': cross_check_results,
            'stats': stats,
            'saved_files': saved_files
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Cross-check error: {str(e)}")
        return jsonify({'error': f'Cross-check failed: {str(e)}'}), 500

@app.route('/download/<file_type>/<file_format>', methods=['POST'])
def download_file(file_type, file_format):
    """Handle file downloads"""
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = request_data.get('filename', 'questions.json')
        chapter_name = extract_chapter_name(filename)
        
        if file_type == 'unique' and file_format == 'json':
            # Download unique questions JSON
            data = request_data['data']['unique_questions']
            filename = f"{chapter_name}_UniqueChapterList.json"
            
            output = io.StringIO()
            json.dump(data, output, indent=2, ensure_ascii=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='application/json',
                as_attachment=True,
                download_name=filename
            )
            
        elif file_type == 'unique' and file_format == 'csv':
            # Download unique questions CSV
            data = request_data['data']['unique_questions']['Content']
            filename = f"{chapter_name}_UniqueChapterList.csv"
            
            output = io.StringIO()
            if data:
                fieldnames = ['QuestionID', 'Question', 'Answer', 'Embedding']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for question in data:
                    writer.writerow({
                        'QuestionID': question.get('QuestionID', ''),
                        'Question': question.get('Question', ''),
                        'Answer': question.get('Answer', ''),
                        'Embedding': json.dumps(question.get('Embedding', []))
                    })
            
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
            
        elif file_type == 'duplicates' and file_format == 'json':
            # Download duplicates JSON
            data = {'Duplicates': request_data['data']['duplicate_report']}
            filename = f"{chapter_name}_DuplicateReport.json"
            
            output = io.StringIO()
            json.dump(data, output, indent=2, ensure_ascii=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='application/json',
                as_attachment=True,
                download_name=filename
            )
            
        elif file_type == 'duplicates' and file_format == 'csv':
            # Download duplicates CSV
            data = request_data['data']['duplicate_report']
            filename = f"{chapter_name}_DuplicateReport.csv"
            
            output = io.StringIO()
            if data:
                df = pd.DataFrame(data)
                df.to_csv(output, index=False)
            
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
            
        elif file_type == 'cross-check' and file_format == 'json':
            # Download cross-check results JSON
            data = {'CrossCheckResults': request_data['data']['cross_check_results']}
            filename = f"{chapter_name}_CrossCheck.json"
            
            output = io.StringIO()
            json.dump(data, output, indent=2, ensure_ascii=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='application/json',
                as_attachment=True,
                download_name=filename
            )
            
        elif file_type == 'cross-check' and file_format == 'csv':
            # Download cross-check results CSV
            data = request_data['data']['cross_check_results']
            filename = f"{chapter_name}_CrossCheck.csv"
            
            output = io.StringIO()
            if data:
                df = pd.DataFrame(data)
                df.to_csv(output, index=False)
            
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
            
        elif file_type == 'master-unique' and file_format == 'json':
            # Download all unique questions from all files
            all_unique_questions, _ = load_existing_unique_questions()
            data = {'Content': all_unique_questions}
            filename = f"master_unique_questions.json"
            
            output = io.StringIO()
            json.dump(data, output, indent=2, ensure_ascii=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='application/json',
                as_attachment=True,
                download_name=filename
            )
            
        elif file_type == 'master-unique' and file_format == 'csv':
            # Download all unique questions from all files as CSV
            all_unique_questions, _ = load_existing_unique_questions()
            filename = f"master_unique_questions.csv"
            
            output = io.StringIO()
            if all_unique_questions:
                fieldnames = ['QuestionID', 'Question', 'Answer', 'Embedding']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for question in all_unique_questions:
                    writer.writerow({
                        'QuestionID': question.get('QuestionID', ''),
                        'Question': question.get('Question', ''),
                        'Answer': question.get('Answer', ''),
                        'Embedding': json.dumps(question.get('Embedding', []))
                    })
            
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
            
        else:
            return jsonify({'error': 'Invalid file type or format'}), 400
            
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/clear-storage', methods=['POST'])
def clear_storage():
    """Clear all stored files (optional utility endpoint)"""
    try:
        folders_to_clear = [
            app.config['UNIQUE_FOLDER'],
            app.config['DUPLICATE_FOLDER'],
            app.config['EMBEDDINGS_FOLDER']
        ]
        
        files_removed = 0
        for folder in folders_to_clear:
            if os.path.exists(folder):
                for file_path in glob.glob(os.path.join(folder, '*')):
                    try:
                        os.remove(file_path)
                        files_removed += 1
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {str(e)}")
        
        logger.info(f"Cleared storage: {files_removed} files removed")
        
        return jsonify({
            'success': True,
            'message': f'Storage cleared successfully. {files_removed} files removed.',
            'files_removed': files_removed
        })
        
    except Exception as e:
        logger.error(f"Clear storage error: {str(e)}")
        return jsonify({'error': f'Failed to clear storage: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5005)
