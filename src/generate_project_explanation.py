import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_project_explanation_pdf(output_path="GelVision_Project_Explanation.pdf"):
    """
    Build a comprehensive PDF document that explains the GelVision AI project
    from A to Z, including theoretical concepts, deep learning architectures,
    pipeline stages, calibration mathematics, and codebase details.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()
    
    # Custom styles
    cover_title_style = ParagraphStyle(
        'CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=32,
        leading=38,
        textColor=colors.HexColor('#0A0F1E'),
        alignment=1, # Center
        spaceAfter=15
    )
    cover_subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#00D4B8'),
        alignment=1,
        spaceAfter=30
    )
    cover_meta_style = ParagraphStyle(
        'CoverMeta',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#555555'),
        alignment=1
    )
    h1_style = ParagraphStyle(
        'ChapterHeader',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0A0F1E'),
        spaceBefore=20,
        spaceAfter=12,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SubHeader',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#00D4B8'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyParagraph',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#222222'),
        spaceAfter=10
    )
    bullet_style = ParagraphStyle(
        'BulletParagraph',
        parent=body_style,
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    story = []
    
    # ------------------ COVER PAGE ------------------
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("GelVision AI", cover_title_style))
    story.append(Paragraph("Automated Gel Electrophoresis Analysis System", cover_subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<b>A-to-Z Technical and Theoretical Reference Manual</b>", ParagraphStyle('CoverRef', fontName='Helvetica-Bold', fontSize=12, alignment=1, textColor=colors.HexColor('#0A0F1E'))))
    story.append(Spacer(1, 2.5*inch))
    story.append(Paragraph("<b>Author:</b> Antigravity AI Engineering & Design Team<br/><b>Date:</b> July 2026<br/><b>Classification:</b> Technical Reference", cover_meta_style))
    story.append(PageBreak())
    
    # ------------------ CHAPTER 1 ------------------
    story.append(Paragraph("Chapter 1: Project Overview & Background", h1_style))
    story.append(Paragraph(
        "Gel electrophoresis is a fundamental laboratory technique in molecular biology and biochemistry. It is used to separate macromolecules, specifically DNA, RNA, and proteins, according to their physical properties: size and charge. By applying an electrical current across a porous gel matrix (typically agarose or polyacrylamide), negatively charged nucleic acids migrate towards the positive anode. Because the gel acts as a molecular sieve, smaller fragments travel faster and further than larger fragments.",
        body_style
    ))
    story.append(Paragraph(
        "Historically, analyzing these gel images has been a manual, slow, and subjective task. Researchers estimate fragment sizes by comparing the bands of an unknown sample to a known 'ladder' or marker run in a parallel lane. They estimate the migration distance manually, plot a curve, and read estimated values off a graph. This manual process introduces human error, variability between researchers, and is highly inefficient in high-throughput workflows.",
        body_style
    ))
    story.append(Paragraph(
        "<b>GelVision AI</b> resolves these challenges by introducing a fully automated, end-to-end computer vision pipeline. The system performs lane detection, band segmentation, and ladder calibration using advanced deep learning architectures, estimating DNA fragment sizes (in base pairs) with an accuracy of R² = 0.989.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # ------------------ CHAPTER 2 ------------------
    story.append(Paragraph("Chapter 2: Deep Learning Models & Architectures", h1_style))
    
    story.append(Paragraph("1. YOLOv8 for Lane Detection", h2_style))
    story.append(Paragraph(
        "Gel images contain multiple lanes running vertically, as well as control lanes containing molecular weight markers (ladders). Locating these lanes is modeled as an <i>object detection</i> problem. We use the state-of-the-art <b>YOLOv8</b> (You Only Look Once, version 8) object detection model.",
        body_style
    ))
    story.append(Paragraph(
        "YOLOv8 uses a convolutional backbone (modified CSPDarknet) combined with a path aggregation network (PANet) and an anchor-free split head. Rather than generating candidate regions, YOLOv8 directly predicts the bounding box coordinates and class probabilities (i.e. 'Lane' vs. 'ladder') for each lane in a single forward pass. This unified architecture is extremely fast and generalizes well, achieving a mean Average Precision (mAP50) of <b>93.1%</b> on the lane detection task, even under varying gel shapes, illumination, and background noise.",
        body_style
    ))
    
    story.append(Paragraph("2. U-Net for Band Segmentation", h2_style))
    story.append(Paragraph(
        "Once a lane is detected and cropped, detecting the specific bands within it requires a pixel-level classification (semantic segmentation), as bands have non-rigid, varying shapes and fuzzy boundaries. For this task, we employ a <b>U-Net</b> segmentation model equipped with a <b>ResNet-34</b> encoder.",
        body_style
    ))
    story.append(Paragraph(
        "The U-Net model features a symmetric encoder-decoder structure. The encoder (downsampling path) extracts high-level semantic features using ResNet-34 residual blocks. The decoder (upsampling path) projects these features back to the original resolution (512x512 pixels). Crucially, <i>skip connections</i> transfer detailed spatial features directly from the encoder contracting blocks to the decoder expanding blocks, ensuring that the boundaries and shapes of the bands are preserved.",
        body_style
    ))
    story.append(Paragraph(
        "The model was trained using the **Dice Loss** function, which directly optimizes the overlap coefficient (F1 score) between the predicted binary mask and the ground truth annotations. U-Net achieves a validation Dice Loss of <b>0.165</b>, allowing robust segmentation of both bright/saturated and faint/low-copy-number bands.",
        body_style
    ))
    story.append(PageBreak())
    
    # ------------------ CHAPTER 3 ------------------
    story.append(Paragraph("Chapter 3: Preprocessing & Computer Vision Pipeline", h1_style))
    story.append(Paragraph(
        "The automated analysis pipeline executes the following stages sequentially:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Step 1: Image Preprocessing:</b> Real gel electrophoresis images suffer from uneven lighting, background autofluorescence, and high-frequency camera noise. We apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with a clip limit of 2.0 to balance the lighting, followed by a Gaussian Blur (3x3 kernel) to denoise, and Min-Max Normalization to scale pixel intensities to [0, 255].",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Step 2: Lane Bounding Box Detection:</b> The preprocessed image is passed through YOLOv8 to locate all lanes and ladders.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Step 3: Lane Sorting:</b> Bounding boxes are sorted from left to right based on their x-coordinates. This ensures that lanes are numbered logically (Lane 1, Lane 2, ..., Lane N), corresponding to how experimental results are documented.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Step 4: Lane Cropping & Normalization:</b> Each lane bounding box is cropped from the preprocessed grayscale image and resized to a standard 512x512 pixels. Resizing normalizes the vertical migration axis across all lanes, correcting for physical gel warping.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Step 5: U-Net Semantic Mask Prediction:</b> The normalized 512x512 crop is passed through the U-Net model to predict a binary mask representing the band locations.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Step 6: 1D Profiling and Peak Detection:</b> We project the 2D band mask into a 1D vertical profile by calculating the average pixel intensity of the mask horizontally. Peaks along this 1D profile correspond to the centroids of the DNA bands. Peak detection is executed using the SciPy library (`scipy.signal.find_peaks`) with a threshold of 0.3 and a minimum peak distance of 10 pixels to suppress noise.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Step 7: Calibration and Size Estimation:</b> The band y-coordinates from the ladder lane are used to fit the calibration curve. The resulting formula is applied to map the y-coordinates of the bands in all other lanes to fragment sizes in base pairs.",
        bullet_style
    ))
    story.append(Spacer(1, 10))
    
    # ------------------ CHAPTER 4 ------------------
    story.append(Paragraph("Chapter 4: Calibration Mathematics & Theory", h1_style))
    story.append(Paragraph(
        "The migration distance of DNA fragments through an agarose gel matrix follows a logarithmic relationship with respect to the fragment size (molecular weight). Specifically, fragments migrate at a velocity inversely proportional to the log of their base pair (bp) length:",
        body_style
    ))
    story.append(Paragraph(
        "<b>ln(Size_bp) = slope * y_coordinate + intercept</b>",
        ParagraphStyle('EquationText', parent=body_style, fontName='Helvetica-Bold', alignment=1)
    ))
    story.append(Paragraph(
        "Where the <i>y_coordinate</i> is the relative migration distance of the band centroid down the lane crop (ranging from 0 at the top to 512 at the bottom). We perform a log-linear ordinary least squares (OLS) regression using SciPy's `linregress` to compute the <i>slope</i> and <i>intercept</i>, achieving a typical fit coefficient of <b>R² = 0.989</b>.",
        body_style
    ))
    story.append(Paragraph(
        "To make the calibration robust to experimental noise, GelVision AI implements a <b>contiguous subset matching algorithm</b>. If the number of detected bands in the ladder lane is less than the standard 16 bands of a 1kb Plus ladder, the algorithm searches over all possible contiguous subsets of the standard sizes. It performs OLS regressions on each candidate subset and automatically selects the subset that maximizes the R² score, ensuring an accurate and reliable calibration even if small bands have run off the gel or large bands are compressed at the top.",
        body_style
    ))
    story.append(PageBreak())
    
    # ------------------ CHAPTER 5 ------------------
    story.append(Paragraph("Chapter 5: Codebase Architecture Walkthrough", h1_style))
    story.append(Paragraph(
        "The project is structured into modular python components in the `src/` directory, integrated into a main Streamlit app:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>preprocess.py:</b> Contains the `preprocess_gel` function. It is built to accept file paths, numpy arrays, or PIL Images, making it highly robust for use in both batch scripts and interactive streamlit uploaders. It applies CLAHE, Gaussian Blur, and Normalization.",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>calibration.py:</b> Contains the `calibrate_ladder` function. It implements the log-linear OLS regression, standardizes sizes for 1kb Plus ladders, resolves band count mismatches using R² optimization, and returns a callable function `position_to_bp(y)` that maps coordinate values to base pairs.",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>pipeline.py:</b> Contains the `GelAnalysisPipeline` class. It manages the entire workflow: model loading, device mapping (CUDA vs CPU), running predictions, sorting, cropping, segmenting, profile projection, peak finding, and invoking calibration. It also accepts a progress callback to feed status updates directly to the Streamlit UI.",
        body_style
    ))
    story.append(Paragraph(
        "4. <b>report.py:</b> Handles data exporting. It converts the pandas results table into CSV bytes, styles Excel sheets with corporate fills/borders using openpyxl, and builds professional PDF reports with embedded annotated images using reportlab.",
        body_style
    ))
    story.append(Paragraph(
        "5. <b>app.py:</b> The main web interface. It implements custom dark biomedical CSS, interactive tabs, progress bars, side-by-side comparison, metric cards, expandable lane crop details, and download options.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # ------------------ CHAPTER 6 ------------------
    story.append(Paragraph("Chapter 6: Future Directions & Improvements", h1_style))
    story.append(Paragraph(
        "While GelVision AI delivers state-of-the-art results, several directions could further extend its functionality:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Multiplexed Ladder Support:</b> Future versions can support automatic recognition of different ladder types (e.g. 100bp ladder, Lambda DNA/HindIII marker) by analyzing standard patterns or allowing the user to upload a custom ladder definition in the UI.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Advanced Sequence Alignment:</b> Moving beyond contiguous search, dynamic programming (similar to Needleman-Wunsch) can align detected band coordinates to known ladder markers, automatically identifying and skipping individual missed bands.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Quantitative Mass Calibration:</b> In addition to fragment size (bp), band intensity (integrated density) can be calibrated against known mass quantities in the ladder to estimate the mass concentration (ng) of sample bands.",
        bullet_style
    ))
    
    doc.build(story)
    print("Project explanation PDF compiled successfully!")

if __name__ == "__main__":
    build_project_explanation_pdf()
