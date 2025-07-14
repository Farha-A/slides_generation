from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
import os
from dotenv import load_dotenv
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display
from io import BytesIO
import time
import urllib.parse
import re
import logging
import threading
import json
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_TIMEOUT'] = 1800

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
CONTENT_FOLDER = os.path.join(BASE_DIR, 'content_text')
GEMINI_FOLDER = os.path.join(BASE_DIR, 'gemini_pdfs')
PROGRESS_FOLDER = os.path.join(BASE_DIR, 'progress')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONTENT_FOLDER'] = CONTENT_FOLDER
app.config['GEMINI_FOLDER'] = GEMINI_FOLDER
app.config['PROGRESS_FOLDER'] = PROGRESS_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for folder in [UPLOAD_FOLDER, CONTENT_FOLDER, GEMINI_FOLDER, PROGRESS_FOLDER]:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {folder}: {e}")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

processing_status = {}
status_lock = threading.Lock()

try:
    FONTS_DIR = os.path.join(BASE_DIR, 'fonts')
    os.makedirs(FONTS_DIR, exist_ok=True)
    pdfmetrics.registerFont(TTFont('Amiri', os.path.join(FONTS_DIR, 'Amiri-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(FONTS_DIR, 'DejaVuSans.ttf')))
except Exception as e:
    logger.error(f"Error registering fonts: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    grade = request.form.get('grade', '').strip()
    course = request.form.get('course', '').strip()
    section = request.form.get('section', '').strip()
    language = request.form.get('language', '').strip()
    country = request.form.get('country', '').strip()

    if file.filename == '' or not file.filename.endswith('.pdf'):
        logger.error("Invalid file or no file selected")
        return jsonify({'error': 'Invalid file or no file selected'}), 400

    if not all([grade, course, section, language, country]):
        logger.error("Missing required form fields in upload")
        return jsonify({'error': 'Missing required form fields'}), 400

    try:
        job_id = f"{int(time.time())}_{secure_filename(file.filename)}"
        original_filename = os.path.splitext(file.filename)[0]
        base_filename = f"{course}_{grade}_{section}_{language}_{country}_{original_filename}"
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.pdf")
        txt_path = os.path.join(app.config['CONTENT_FOLDER'], f"{base_filename}.txt")
        
        logger.info(f"Saving uploaded file: {file.filename}")
        update_progress(job_id, 'uploading', 5, "Saving uploaded file...")
        
        file.save(pdf_path)
        
        thread = threading.Thread(
            target=process_pdf_background,
            args=(pdf_path, txt_path, grade, course, section, language, country, original_filename, job_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'File uploaded successfully. Processing started.'
        })
        
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/progress/<job_id>')
def progress(job_id):
    def generate():
        last_update = None
        start_time = time.time()
        timeout = 1800
        while time.time() - start_time < timeout:
            with status_lock:
                if job_id in processing_status:
                    current_status = processing_status[job_id]
                    if current_status != last_update:
                        last_update = current_status
                        yield f"data: {json.dumps(current_status)}\n\n"
                    if current_status['stage'] in ['completed', 'error']:
                        break
                else:
                    yield f"data: {json.dumps({'stage': 'initializing', 'progress': 0, 'message': 'Initializing job...', 'timestamp': datetime.now().isoformat()})}\n\n"
                    time.sleep(1)
                    continue
            time.sleep(1)
        if time.time() - start_time >= timeout:
            yield f"data: {json.dumps({'stage': 'error', 'progress': 0, 'message': 'Progress tracking timed out', 'timestamp': datetime.now().isoformat()})}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/get_content_files')
def get_content_files():
    try:
        files = [f for f in os.listdir(app.config['CONTENT_FOLDER']) if f.endswith('.txt')]
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error listing content files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_generated_slide_points')
def get_generated_slide_points():
    try:
        files = [f for f in os.listdir(app.config['GEMINI_FOLDER']) if f.endswith('.pdf')]
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error listing generated slide points: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_text_file/<filename>')
def view_text_file(filename):
    decoded_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(app.config['CONTENT_FOLDER'], decoded_filename)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        logger.error(f"Text file not found: {file_path}")
        return jsonify({'error': 'Text file not found'}), 404
    except Exception as e:
        logger.error(f"Error serving text file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_text_as_pdf/<filename>')
def download_text_as_pdf(filename):
    decoded_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(app.config['CONTENT_FOLDER'], decoded_filename)
    try:
        if not os.path.exists(file_path):
            logger.error(f"Text file not found: {file_path}")
            return jsonify({'error': 'Text file not found'}), 404

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pdf_path = os.path.join(app.config['CONTENT_FOLDER'], f"{os.path.splitext(decoded_filename)[0]}_converted.pdf")
        generate_pdf_from_text(content, pdf_path, language='english')

        if not os.path.exists(pdf_path):
            logger.error(f"PDF not generated: {pdf_path}")
            return jsonify({'error': 'PDF generation failed'}), 500

        response = send_file(pdf_path, mimetype='application/pdf', as_attachment=True)
        
        try:
            os.remove(pdf_path)
            logger.info(f"Cleaned up generated PDF: {pdf_path}")
        except Exception as cleanup_error:
            logger.warning(f"Could not delete generated PDF {pdf_path}: {cleanup_error}")

        return response
    except Exception as e:
        logger.error(f"Error serving PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_pdf/<filename>')
def view_pdf(filename):
    decoded_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(app.config['GEMINI_FOLDER'], decoded_filename)
    try:
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='application/pdf', as_attachment=True)
        logger.error(f"PDF not found: {file_path}")
        return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        logger.error(f"Error serving PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_slides', methods=['POST'])
def generate_slides():
    filename = request.form.get('filename', '').strip()
    grade = request.form.get('grade', '').strip()
    course = request.form.get('course', '').strip()
    section = request.form.get('section', '').strip()
    country = request.form.get('country', '').strip()
    language = request.form.get('language', '').strip()
    original_filename = os.path.splitext(filename)[0]
    
    # if not all([filename, grade, course, section, country, language]):
    #     logger.error("Missing required form fields in generate_slides")
    #     return jsonify({'error': 'Missing required form fields'}), 400
    
    txt_path = os.path.join(app.config['CONTENT_FOLDER'], filename)
    if not os.path.exists(txt_path):
        logger.error(f"Text file not found: {txt_path}")
        return jsonify({'error': 'Text file not found'}), 404
    
    # base_filename = f"{course}_{grade}_{section}_{language}_{country}_{original_filename}"
    output_txt_path = os.path.join(app.config['GEMINI_FOLDER'], f"{original_filename}_gemini_response.txt")
    pdf_path = os.path.join(app.config['GEMINI_FOLDER'], f"{original_filename}_gemini_response.pdf")
    job_id = f"gen_{int(time.time())}_{secure_filename(filename)}"
    
    update_progress(job_id, 'initializing', 0, "Initializing slide generation...")
    
    def process_slides_background():
        try:
            update_progress(job_id, 'initializing', 5, "Preparing to process content...")
            
            total_pages = get_page_count(txt_path)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = construct_prompt(grade, course, section, country, language)
            
            update_progress(job_id, 'reading', 10, "Reading content file...")
            with open(txt_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
            
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write("")
            
            logger.info(f"Processing {total_pages} pages for Gemini...")
            update_progress(job_id, 'processing', 20, f"Processing {total_pages} pages...")
            
            if total_pages <= 20:
                response = model.generate_content(prompt + "\n\nContent:\n" + full_content)
                if response and response.text:
                    with open(output_txt_path, 'a', encoding='utf-8') as f:
                        f.write(response.text + "\n")
                else:
                    logger.error("No response from Gemini or empty response")
                    update_progress(job_id, 'error', 0, "No response from Gemini")
                    if os.path.exists(output_txt_path):
                        os.remove(output_txt_path)
                    return
            else:
                pages = re.split(r'--- Page \d+ ---', full_content)[1:]
                batch_size = 15
                batch_count = 0
                
                for start_page in range(0, total_pages, batch_size):
                    end_page = min(start_page + batch_size, total_pages)
                    batch_count += 1
                    logger.info(f"Processing batch {batch_count}: pages {start_page + 1}-{end_page}")
                    update_progress(job_id, 'processing', 20 + (batch_count / (total_pages / batch_size)) * 60, 
                                  f"Processing pages {start_page + 1}-{end_page}...")
                    
                    page_content = "".join(pages[start_page:end_page])
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = model.generate_content(prompt + "\n\nContent:\n" + page_content)
                            if response and response.text:
                                with open(output_txt_path, 'a', encoding='utf-8') as f:
                                    f.write(response.text + "\n")
                                break
                            else:
                                logger.warning(f"No response for pages {start_page + 1}-{end_page}, attempt {attempt + 1}")
                                if attempt < max_retries - 1:
                                    time.sleep(2 ** attempt)
                        except Exception as api_error:
                            logger.error(f"API error for pages {start_page + 1}-{end_page}, attempt {attempt + 1}: {api_error}")
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)
                            else:
                                logger.error(f"Failed to process pages {start_page + 1}-{end_page} after {max_retries} attempts")
                    
                    time.sleep(1)
            
            update_progress(job_id, 'verifying', 80, "Verifying processed content...")
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                response_text = f.read()
            
            if not response_text.strip():
                logger.error(f"Output file is empty: {output_txt_path}")
                update_progress(job_id, 'error', 0, "Empty response from Gemini")
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                return
            logger.info("Generating PDF from Gemini response...")
            update_progress(job_id, 'generating_pdf', 90, "Generating PDF...")
            generate_pdf_from_text(response_text, pdf_path, language=language)
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF not generated: {pdf_path}")
                update_progress(job_id, 'error', 0, "PDF generation failed")
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                return
            
            try:
                os.remove(output_txt_path)
                logger.info(f"Successfully cleaned up intermediate text file: {output_txt_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not delete intermediate text file {output_txt_path}: {cleanup_error}")
            
            logger.info(f"Process completed successfully. PDF available at: {pdf_path}")
            update_progress(job_id, 'completed', 100, "Slide points generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating slides: {e}")
            update_progress(job_id, 'error', 0, f"Error: {str(e)}")
            try:
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                    logger.info(f"Cleaned up text file after error: {output_txt_path}")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    logger.info(f"Cleaned up partial PDF file after error: {pdf_path}")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
    
    thread = threading.Thread(target=process_slides_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'job_id': job_id, 'message': 'Slide generation started'})

def update_progress(job_id, stage, progress, message=""):
    with status_lock:
        processing_status[job_id] = {
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    
    try:
        progress_file = os.path.join(app.config['PROGRESS_FOLDER'], f"{job_id}.json")
        with open(progress_file, 'w') as f:
            json.dump(processing_status[job_id], f)
    except Exception as e:
        logger.error(f"Error saving progress to file: {e}")

def has_extractable_text(pdf_file):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages_to_check = min(3, len(reader.pages))
            
            for i in range(pages_to_check):
                text = reader.pages[i].extract_text() or ''
                if len(text.strip()) > 10:
                    return True
    except Exception as e:
        logger.error(f"Error checking extractable text for {pdf_file}: {e}")
    return False

def extract_text_streaming(pdf_file, output_path, start_page, end_page, job_id=None):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            mode = 'w' if start_page == 0 else 'a'
            
            with open(output_path, mode, encoding='utf-8') as f:
                total_pages = end_page - start_page
                for i in range(start_page, min(end_page, len(reader.pages))):
                    if job_id:
                        progress = ((i - start_page + 1) / total_pages) * 100
                        update_progress(job_id, 'extracting', progress, f"Extracting page {i + 1}")
                    
                    text = reader.pages[i].extract_text() or ''
                    f.write(f"\n--- Page {i + 1} ---\n")
                    f.write(text)
                    f.write("\n")
                    f.flush()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_file}: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error extracting text: {e}\n")

def ocr_content_streaming(pdf_path, output_path, start_page, end_page, language='eng', job_id=None):
    try:
        logger.info(f"Starting OCR for pages {start_page+1} to {end_page} (language: {language})")
        
        batch_size = 1
        mode = 'w' if start_page == 0 else 'a'
        
        with open(output_path, mode, encoding='utf-8') as f:
            total_pages = end_page - start_page
            
            for batch_start in range(start_page, end_page, batch_size):
                batch_end = min(batch_start + batch_size, end_page)
                
                try:
                    images = convert_from_path(
                        pdf_path,
                        dpi=100,
                        first_page=batch_start+1,
                        last_page=batch_end,
                        fmt='jpeg',
                        jpegopt={'quality': 80, 'progressive': True, 'optimize': True}
                    )
                    
                    for i, image in enumerate(images, start=batch_start):
                        if job_id:
                            progress = ((i - start_page + 1) / total_pages) * 100
                            update_progress(job_id, 'ocr', progress, f"OCR processing page {i + 1}")
                        
                        logger.info(f"Processing page {i + 1} with OCR...")
                        
                        max_dimension = 1800
                        if max(image.size) > max_dimension:
                            ratio = max_dimension / max(image.size)
                            new_size = tuple(int(dim * ratio) for dim in image.size)
                            image = image.resize(new_size, Image.Resampling.LANCZOS)
                        
                        image = image.convert('L')
                        
                        try:
                            custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:-()[]{}'
                            if language == 'ara':
                                custom_config = r'--oem 3 --psm 6'
                            
                            text = pytesseract.image_to_string(
                                image,
                                lang=language,
                                config=custom_config,
                                timeout=180
                            )
                            
                            f.write(f"\n--- Page {i + 1} ---\n")
                            f.write(text)
                            f.write("\n")
                            f.flush()
                            
                        except Exception as page_error:
                            logger.error(f"OCR error on page {i + 1}: {page_error}")
                            f.write(f"\n--- Page {i + 1} (OCR Error) ---\n")
                            f.write(f"Error processing page: {str(page_error)}\n")
                            f.write("\n")
                        
                        del image
                    
                    del images
                    
                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_start+1}-{batch_end}: {batch_error}")
                    f.write(f"\n--- Pages {batch_start+1}-{batch_end} (Batch Error) ---\n")
                    f.write(f"Error: {str(batch_error)}\n")
                
                time.sleep(0.1)
                
        logger.info(f"OCR completed for pages {start_page+1} to {end_page}")
        
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error performing OCR on pages {start_page+1}-{end_page}: {e}\n")

def process_chunk_with_progress(pdf_path, txt_path, start_page, end_page, is_extractable, language, job_id=None):
    try:
        if is_extractable:
            extract_text_streaming(pdf_path, txt_path, start_page, end_page, job_id)
        else:
            ocr_content_streaming(pdf_path, txt_path, start_page, end_page, language=language, job_id=job_id)
    except Exception as chunk_error:
        logger.error(f"Error processing chunk {start_page}-{end_page}: {chunk_error}")
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Pages {start_page + 1}-{end_page} (Processing Error) ---\n")
            f.write(f"Error: {str(chunk_error)}\n")

def process_pdf_background(pdf_path, txt_path, grade, course, section, language, country, original_filename, job_id):
    try:
        update_progress(job_id, 'analyzing', 10, "Analyzing PDF structure...")
        
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
        
        logger.info(f"PDF has {num_pages} pages")
        is_extractable = has_extractable_text(pdf_path)
        logger.info(f"PDF has extractable text: {is_extractable}")
        
        update_progress(job_id, 'processing', 20, f"Processing {num_pages} pages...")
        
        chunk_size = 1 if not is_extractable and num_pages > 100 else (2 if not is_extractable else 8)
        
        for start in range(0, num_pages, chunk_size):
            end = min(start + chunk_size, num_pages)
            
            progress = 20 + ((start / num_pages) * 60)
            update_progress(job_id, 'processing', progress, f"Processing pages {start+1}-{end}")
            
            process_chunk_with_progress(pdf_path, txt_path, start, end, is_extractable, language, job_id)
        
        update_progress(job_id, 'verifying', 85, "Verifying processed content...")
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_pages = []
        for page_num in range(1, num_pages + 1):
            if f"--- Page {page_num} ---" not in content:
                missing_pages.append(page_num)
        
        if missing_pages:
            logger.warning(f"Reprocessing missing pages: {missing_pages}")
            update_progress(job_id, 'reprocessing', 90, f"Reprocessing {len(missing_pages)} missing pages...")
            
            for page_num in missing_pages:
                process_chunk_with_progress(pdf_path, txt_path, page_num - 1, page_num, is_extractable, language, job_id)
        
        update_progress(job_id, 'completed', 100, "Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
        update_progress(job_id, 'error', 0, f"Error: {str(e)}")
        
    finally:
        with status_lock:
            if job_id in processing_status:
                del processing_status[job_id]
        progress_file = os.path.join(app.config['PROGRESS_FOLDER'], f"{job_id}.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)

def generate_pdf_from_text(text, output_path, language='english'):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        story = []
        lines = text.split('\n')
        
        arabic_style = ParagraphStyle(
            name='Arabic',
            fontName='Amiri',
            fontSize=12,
            leading=16,
            alignment=2,
            spaceAfter=12,
            textColor=colors.black,
            allowWidows=1,
            allowOrphans=1,
            splitLongWords=False,
            wordWrap='RTL'
        )
        
        english_style = ParagraphStyle(
            name='English',
            fontName='DejaVuSans',
            fontSize=12,
            leading=16,
            alignment=1,
            spaceAfter=12,
            textColor=colors.black
        )
        
        for line in lines:
            if not line.strip():
                story.append(Spacer(1, 6))
                continue
            has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in line)
            has_english = any(c.isascii() and c.isalpha() for c in line)
            
            if has_arabic:
                try:
                    reshaped_text = arabic_reshaper.reshape(line)
                    bidi_text = get_display(reshaped_text)
                except Exception as e:
                    logger.error(f"Error reshaping Arabic text: {e}")
                    bidi_text = line
                
                if has_english:
                    parts = re.split(r'([.!?:;،؛])', bidi_text)
                    for part in parts:
                        if part.strip():
                            part_has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in part)
                            style = arabic_style if part_has_arabic else english_style
                            p = Paragraph(part.strip(), style)
                            story.append(p)
                else:
                    p = Paragraph(bidi_text, arabic_style)
                    story.append(p)
            else:
                p = Paragraph(line, english_style)
                story.append(p)
            
            story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buffer.read())
        
        buffer.close()
        
    except Exception as e:
        logger.error(f"Error generating PDF for {language}: {e}")
        raise

def construct_prompt(grade, course, section, country, language):
    return f"""
Role:
You are an expert Instructional Content Designer specializing in creating visually organized, curriculum-aligned slide presentations for classroom use.

Objective:
Generate a well-structured, age-appropriate slide presentation for:
Grade: {grade}
Course: {course}
Curriculum Section: {section}
Country: {country}
Language: {language}

Slide Generation Instructions:
Content Structure:
Create content based on typical curriculum for {course}, {section} for Grade {grade} in {country}.
Include key ideas, definitions, explanations, examples, and terms.
Use as many slides as needed for clarity—no limit.
If content includes multiple lessons or sub-lessons, generate:
A separate slide set for each.
Reset slide numbers at the start of each.
Label each lesson/sub-lesson clearly.
Concatenate all outputs into one continuous document.

Each Slide Must Include:
Slide Title: A clear and concise heading for the slide's main idea.
Bullet Points: 3–5 simplified, student-friendly bullets summarizing key information.
Suggested Visual: Description of a diagram, image, illustration, or chart that supports understanding.
Optional Think Prompt: A short reflective or analytical question aligned to Bloom's Taxonomy.
Numbering & Labeling Instructions:
Use standard chapter and lesson numbering for {course}, {section} in {country}.
Do not continue lesson numbers across chapters.
Always label slides clearly using this structure:
Chapter X – Lesson X.Y – Slide Z
Reset the slide counter for each new lesson.
Preserve standard curriculum-based numbering exactly (e.g., 1.1, 1.2... 2.1, 2.2, etc.).

Slide Style:
Use {language} at a reading level appropriate for Grade {grade}.
Use clear, engaging, instructive language.
Follow the "one concept per slide" principle—avoid text overload.

Curriculum & Learning Progression Guidelines:
Curriculum Alignment:
Structure content in line with the learning objectives and flow of the {section} curriculum of {country}.
Integrate subject-specific terminology and grade-appropriate academic language.

Bloom's Taxonomy Integration:
Start with slides that promote Remembering and Understanding.
Progress into Applying and Analyzing.
If appropriate, conclude with tasks that support Evaluating or Creating (e.g., student reflection, real-world problem-solving).

Output Format:
Number each slide (Slide 1, Slide 2, etc.).
Restart numbering for each new lesson or sub-lesson.
Keep format and tone consistent across all generated sets.
Remove all markdown style formatting (e.g., no bold, italics, etc.).
"""

def get_page_count(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        page_numbers = re.findall(r'--- Page \d+ ---', content)
        return len(page_numbers) if page_numbers else 0
    except Exception as e:
        logger.error(f"Error reading page count: {e}")
        return 0

if __name__ == '__main__':
    app.run(debug=True, port=1000)