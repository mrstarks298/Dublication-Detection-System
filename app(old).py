from flask import Flask, request, jsonify, render_template, send_file
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import time
from werkzeug.utils import secure_filename
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
import shutil
import glob

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STORAGE_FOLDER'] = 'question_storage'
app.config['UNIQUE_FOLDER'] = os.path.join(app.config['STORAGE_FOLDER'], 'unique_questions')
app.config['DUPLICATE_FOLDER'] = os.path.join(app.config['STORAGE_FOLDER'], 'duplicate_reports')

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['STORAGE_FOLDER'], 
               app.config['UNIQUE_FOLDER'], app.config['DUPLICATE_FOLDER']]:
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

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return cosine_similarity([embedding1], [embedding2])[0][0]

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
    unique_files = glob.glob(os.path.join(app.config['UNIQUE_FOLDER'], '*_unique.json'))
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

def find_cross_file_duplicates(current_questions, existing_questions, thresholds):
    """Find duplicates between current file and existing unique questions (Questions only)"""
    logger.info(f"Comparing {len(current_questions)} current questions with {len(existing_questions)} existing questions")
    
    if not existing_questions:
        return []
    
    # Generate embeddings for existing questions
    logger.info("Generating embeddings for existing questions...")
    existing_question_embeddings = []
    
    for question in existing_questions:
        existing_question_embeddings.append(encode_text(question.get('Question', '')))
    
    # Generate embeddings for current questions
    logger.info("Generating embeddings for current questions...")
    current_question_embeddings = []
    
    for question in current_questions:
        current_question_embeddings.append(encode_text(question.get('Question', '')))
    
    # Find cross-file duplicates (Questions only)
    cross_duplicates = []
    questions_to_remove = set()  # Question IDs from current file to remove
    
    for i, current_q in enumerate(current_questions):
        for j, existing_q in enumerate(existing_questions):
            # Question similarity
            q_sim = calculate_cosine_similarity(
                current_question_embeddings[i], 
                existing_question_embeddings[j]
            )
            
            if q_sim >= thresholds['semantic_duplicate_threshold']:
                tag = "Near Duplicate" if q_sim >= thresholds['near_duplicate_threshold'] else "Semantic Duplicate"
                cross_duplicates.append({
                    'OriginalQuestionID': existing_q['QuestionID'],  # Existing question is original
                    'DuplicateQuestionID': current_q['QuestionID'],   # Current question is duplicate
                    'OriginalQuestion': existing_q.get('Question', ''),
                    'DuplicateQuestion': current_q.get('Question', ''),
                    'SimilarityScore': float(q_sim),
                    'Tag': tag,
                    'Type': 'Question',
                    'CrossFile': True
                })
                questions_to_remove.add(current_q['QuestionID'])
                break  # Don't check this current question against other existing questions
    
    logger.info(f"Found {len(cross_duplicates)} cross-file duplicates")
    logger.info(f"Removing {len(questions_to_remove)} questions from current file")
    
    return cross_duplicates, questions_to_remove

def process_internal_duplicates(data, thresholds):
    """Process duplicates within the current file (Questions only)"""
    logger.info(f"Processing internal duplicates in {len(data)} questions...")
    
    # Generate embeddings for questions only
    logger.info("Generating question embeddings...")
    question_embeddings = []
    
    for i, item in enumerate(data):
        question_text = item.get('Question', '')
        question_embeddings.append(encode_text(question_text))
        
        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(data)} embeddings")
    
    # Find internal duplicates (Questions only)
    logger.info("Finding internal duplicates...")
    duplicates = []
    n = len(data)
    
    # Initialize Union-Find for grouping
    uf = UnionFind(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Question similarity
            q_sim = calculate_cosine_similarity(question_embeddings[i], question_embeddings[j])
            if q_sim >= thresholds['semantic_duplicate_threshold']:
                tag = "Near Duplicate" if q_sim >= thresholds['near_duplicate_threshold'] else "Semantic Duplicate"
                duplicates.append({
                    'OriginalQuestionID': data[i]['QuestionID'],
                    'DuplicateQuestionID': data[j]['QuestionID'],
                    'OriginalQuestion': data[i].get('Question', ''),
                    'DuplicateQuestion': data[j].get('Question', ''),
                    'SimilarityScore': float(q_sim),
                    'Tag': tag,
                    'Type': 'Question',
                    'CrossFile': False
                })
                # Union the indices for grouping
                uf.union(i, j)
    
    # Get duplicate groups using Union-Find
    logger.info("Creating unique questions list using Union-Find grouping...")
    duplicate_groups = uf.get_groups()
    
    # Track which indices to keep (first from each group)
    indices_to_keep = set(range(n))  # Start with all indices
    
    for group in duplicate_groups:
        # Sort group by original order and keep only the first one
        group.sort()  # Keep the first occurrence (smallest index)
        # Remove all except the first
        for idx in group[1:]:
            indices_to_keep.discard(idx)
    
    # Create unique questions list
    unique_data = [data[i] for i in sorted(indices_to_keep)]
    
    logger.info(f"Found {len(duplicates)} internal duplicate pairs")
    logger.info(f"Found {len(duplicate_groups)} internal duplicate groups")
    logger.info(f"Reduced from {len(data)} to {len(unique_data)} questions after internal deduplication")
    
    return unique_data, duplicates

def save_results(filename, unique_questions, all_duplicates, stats):
    """Save unique questions and duplicate reports to storage folders"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(filename)[0]
    
    # Save unique questions
    unique_filename = f"{base_name}_{timestamp}_unique.json"
    unique_path = os.path.join(app.config['UNIQUE_FOLDER'], unique_filename)
    
    unique_data = {
        'Content': unique_questions,
        'Metadata': {
            'original_filename': filename,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'total_questions': stats.get('total_questions', 0),
            'unique_questions': len(unique_questions),
            'processing_time': stats.get('processing_time', 0)
        }
    }
    
    with open(unique_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    # Save duplicate report
    duplicate_filename = f"{base_name}_{timestamp}_duplicates.json"
    duplicate_path = os.path.join(app.config['DUPLICATE_FOLDER'], duplicate_filename)
    
    duplicate_data = {
        'Duplicates': all_duplicates,
        'Metadata': {
            'original_filename': filename,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'total_duplicates': len(all_duplicates),
            'internal_duplicates': len([d for d in all_duplicates if not d.get('CrossFile', False)]),
            'cross_file_duplicates': len([d for d in all_duplicates if d.get('CrossFile', False)])
        }
    }
    
    with open(duplicate_path, 'w', encoding='utf-8') as f:
        json.dump(duplicate_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results: {unique_filename} and {duplicate_filename}")
    return unique_filename, duplicate_filename

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index(old).html')

@app.route('/storage-info')
def storage_info():
    """Get information about stored files"""
    try:
        unique_files = glob.glob(os.path.join(app.config['UNIQUE_FOLDER'], '*_unique.json'))
        duplicate_files = glob.glob(os.path.join(app.config['DUPLICATE_FOLDER'], '*_duplicates.json'))
        
        storage_stats = {
            'unique_files_count': len(unique_files),
            'duplicate_files_count': len(duplicate_files),
            'total_unique_questions': 0,
            'storage_folder': app.config['STORAGE_FOLDER']
        }
        
        # Count total unique questions
        for file_path in unique_files:
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
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_questions():
    """Process questions for duplicates with cross-file comparison"""
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
        
        logger.info(f"Processing {filename} with thresholds: {thresholds}")
        
        start_time = time.time()
        
        # Step 1: Process internal duplicates within current file
        unique_questions_internal, internal_duplicates = process_internal_duplicates(data, thresholds)
        
        # Step 2: Load existing unique questions from storage
        existing_unique_questions, file_mapping = load_existing_unique_questions()
        
        # Step 3: Find cross-file duplicates
        cross_duplicates = []
        questions_to_remove_cross = set()
        
        if existing_unique_questions:
            cross_duplicates, questions_to_remove_cross = find_cross_file_duplicates(
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
            'existing_files_checked': len(file_mapping)
        }
        
        # Step 7: Save results to storage
        unique_filename, duplicate_filename = save_results(
            filename, final_unique_questions, all_duplicates, stats
        )
        
        result = {
            'unique_questions': {'Content': final_unique_questions},
            'duplicate_report': all_duplicates,
            'stats': stats,
            'saved_files': {
                'unique_filename': unique_filename,
                'duplicate_filename': duplicate_filename
            }
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<file_type>', methods=['POST'])
def download_file(file_type):
    """Generate and serve download files"""
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data provided'}), 400
        
        if file_type == 'unique':
            data = request_data['data']['unique_questions']
            filename = 'unique_questions.json'
        elif file_type == 'duplicates':
            data = request_data['data']['duplicate_report']
            filename = 'duplicate_report.json'
        elif file_type == 'master-unique':
            # Download all unique questions combined
            all_unique, _ = load_existing_unique_questions()
            data = {'Content': all_unique}
            filename = 'master_unique_questions.json'
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=filename,
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time(),
        'storage_folders': {
            'unique': app.config['UNIQUE_FOLDER'],
            'duplicates': app.config['DUPLICATE_FOLDER']
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5003)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
