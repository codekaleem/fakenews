"""
Generate a PDF documentation for the DS Project
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

def generate_pdf():
    # Create PDF document
    pdf_file = "PROJECT_DOCUMENTATION.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2e5c92'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14
    )
    
    # Title
    elements.append(Paragraph("DS PROJECT DOCUMENTATION", title_style))
    elements.append(Paragraph(f"<b>Generated: {datetime.now().strftime('%B %d, %Y')}</b>", 
                             ParagraphStyle('Date', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10)))
    elements.append(Spacer(1, 0.3*inch))
    
    # 1. Libraries Used
    elements.append(Paragraph("1. LIBRARIES USED & PURPOSE", heading_style))
    
    libraries_data = [
        ['Library', 'Version', 'Purpose'],
        ['pandas', 'Latest', 'Data manipulation and analysis. Used for loading, cleaning, and processing CSV datasets containing news articles and labels.'],
        ['numpy', 'Latest', 'Numerical computing library. Provides support for large, multi-dimensional arrays and mathematical operations used in data processing.'],
        ['scikit-learn', 'Latest', 'Machine learning library. Used for TfidfVectorizer (text feature extraction) and PassiveAggressiveClassifier (model training and prediction).'],
        ['nltk', 'Latest', 'Natural Language Processing. Used for downloading stopwords and text preprocessing (removing common English words).'],
        ['streamlit', 'Latest', 'Web application framework. Builds interactive user interface for the fake news detection application with real-time predictions.'],
        ['pillow (PIL)', 'Latest', 'Image processing library. Analyzes images for suspicion scores by computing entropy, color statistics, and aspect ratios.'],
    ]
    
    lib_table = Table(libraries_data, colWidths=[1.2*inch, 0.8*inch, 3.7*inch])
    lib_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c92')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    elements.append(lib_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # 2. Machine Learning Model
    elements.append(Paragraph("2. MACHINE LEARNING MODEL", heading_style))
    
    model_text = """
    <b>Model Type:</b> PassiveAggressiveClassifier<br/>
    <b>Source:</b> scikit-learn library<br/>
    <b>Purpose:</b> Binary classification for fake news detection<br/>
    <br/>
    <b>Why PassiveAggressiveClassifier?</b><br/>
    • Efficient for large-scale text classification tasks<br/>
    • Suitable for online learning with streaming data<br/>
    • Fast training and prediction times<br/>
    • Memory-efficient compared to other algorithms like SVM<br/>
    • Works well with TF-IDF vectorized text features<br/>
    <br/>
    <b>Model Configuration:</b><br/>
    • Max iterations: 50<br/>
    • Random state: 42 (for reproducibility)<br/>
    <br/>
    <b>Feature Extraction:</b><br/>
    • TfidfVectorizer with max_features=5000<br/>
    • N-gram range: (1, 2) - unigrams and bigrams<br/>
    • Sublinear TF scaling for better feature representation<br/>
    <br/>
    <b>Training Data Split:</b><br/>
    • Training set: 80%<br/>
    • Test set: 20%<br/>
    • Stratified split to maintain class distribution<br/>
    """
    
    elements.append(Paragraph(model_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Page Break
    elements.append(PageBreak())
    
    # 3. Text Preprocessing
    elements.append(Paragraph("3. TEXT PREPROCESSING PIPELINE", heading_style))
    
    preprocess_text = """
    The <b>clean_text()</b> function in preprocess.py performs the following steps:<br/>
    <br/>
    1. <b>Lowercase Conversion:</b> Convert all text to lowercase for uniform comparison<br/>
    2. <b>URL Removal:</b> Remove HTTP/HTTPS URLs and www links<br/>
    3. <b>HTML Tag Removal:</b> Strip HTML tags from text<br/>
    4. <b>Special Character Removal:</b> Keep only alphanumeric characters and spaces<br/>
    5. <b>Whitespace Normalization:</b> Replace multiple spaces with single space<br/>
    6. <b>Stopword Removal:</b> Remove common English words using NLTK corpus<br/>
    7. <b>Tokenization:</b> Split cleaned text into individual tokens<br/>
    <br/>
    <b>NLTK Resources:</b> The application automatically downloads English stopwords on first run.
    """
    
    elements.append(Paragraph(preprocess_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # 4. Image Analysis
    elements.append(Paragraph("4. IMAGE ANALYSIS FEATURES", heading_style))
    
    image_text = """
    The <b>image_utils.py</b> module analyzes images to compute suspicion scores based on visual characteristics:<br/>
    <br/>
    <b>Features Extracted:</b><br/>
    • <b>Entropy:</b> Measures randomness/disorder in image (lower values = more uniform, potentially suspicious)<br/>
    • <b>Unique Colors:</b> Number of distinct colors in image<br/>
    • <b>Standard Deviation:</b> Variation in pixel values across color channels<br/>
    • <b>Aspect Ratio:</b> Width-to-height ratio to detect distorted images<br/>
    <br/>
    <b>Scoring Logic:</b><br/>
    • Base score: 0.2<br/>
    • Low entropy (&lt;4.0) adds 0.25 to score (indicates unnatural image)<br/>
    • Medium entropy (4.0-5.0) adds 0.1 to score<br/>
    • Images with &lt;256 unique colors receive additional suspicion points
    """
    
    elements.append(Paragraph(image_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # 5. Database
    elements.append(Paragraph("5. DATABASE USAGE", heading_style))
    
    database_text = """
    <b>Database Type:</b> CSV Files (Comma-Separated Values)<br/>
    <b>Why CSV and Not a Relational Database?</b><br/>
    <br/>
    • <b>Simplicity:</b> CSV files are lightweight and easy to work with for machine learning projects<br/>
    • <b>Data Science Workflow:</b> pandas library natively supports CSV I/O operations<br/>
    • <b>Deployment:</b> Easier to deploy on platforms like Heroku (as indicated by Procfile)<br/>
    • <b>Version Control:</b> CSV files can be tracked in Git for data versioning<br/>
    • <b>Portability:</b> CSV is a universal format, compatible with any tool or language<br/>
    • <b>Dataset Size:</b> Training dataset fits in memory; database overhead not needed<br/>
    <br/>
    <b>Dataset Files:</b><br/>
    • <b>Primary:</b> data/fake_news_dataset.csv<br/>
    • <b>Archive:</b> Additional CSV files in 'archive/' directory for extended training<br/>
    <br/>
    <b>Data Structure:</b><br/>
    • <b>news/text column:</b> Article content<br/>
    • <b>title column:</b> Article headline (optional, concatenated with text)<br/>
    • <b>label column:</b> Binary classification (0=Fake, 1=Real)<br/>
    """
    
    elements.append(Paragraph(database_text, body_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer
    elements.append(PageBreak())
    elements.append(Paragraph("PROJECT SUMMARY", heading_style))
    
    summary_text = """
    This <b>Fake News Detection</b> project is a machine learning application that combines text analysis and image processing to identify fake news articles.<br/>
    <br/>
    <b>Workflow:</b><br/>
    1. Load news dataset from CSV files<br/>
    2. Preprocess text using NLTK and regex techniques<br/>
    3. Extract features using TfidfVectorizer<br/>
    4. Train PassiveAggressiveClassifier model<br/>
    5. Deploy as Streamlit web application<br/>
    6. Accept user input (text and/or images) for real-time predictions<br/>
    7. Return classification results and suspicion scores<br/>
    <br/>
    <b>Key Technologies Stack:</b><br/>
    • <b>Backend:</b> Python with scikit-learn for ML<br/>
    • <b>Frontend:</b> Streamlit for interactive UI<br/>
    • <b>Data Processing:</b> pandas and numpy<br/>
    • <b>NLP:</b> NLTK for text preprocessing<br/>
    • <b>Image Processing:</b> Pillow for image analysis<br/>
    • <b>Deployment:</b> Heroku-ready (Procfile included)<br/>
    """
    
    elements.append(Paragraph(summary_text, body_style))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ PDF generated successfully: {pdf_file}")

if __name__ == "__main__":
    generate_pdf()
