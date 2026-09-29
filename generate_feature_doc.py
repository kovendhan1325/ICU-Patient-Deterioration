"""Generate Feature_List_Priority.docx for the ICU project."""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()

# -- Page margins --
for section in doc.sections:
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

# ── Helper functions ──
def add_heading_styled(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
    return h

def add_table_row(table, cells_data, bold=False, bg_color=None):
    row = table.add_row()
    for i, text in enumerate(cells_data):
        cell = row.cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(str(text))
        run.font.size = Pt(9)
        if bold:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        if bg_color:
            shading = cell._element.get_or_add_tcPr()
            shading_elm = shading.makeelement(qn('w:shd'), {
                qn('w:fill'): bg_color,
                qn('w:val'): 'clear'
            })
            shading.append(shading_elm)
    return row

def make_table(headers, rows, header_color='1A1A2E'):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hdr = table.rows[0]
    for i, text in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shading = cell._element.get_or_add_tcPr()
        shading_elm = shading.makeelement(qn('w:shd'), {
            qn('w:fill'): header_color,
            qn('w:val'): 'clear'
        })
        shading.append(shading_elm)
    # Data rows
    for row_data in rows:
        add_table_row(table, row_data)
    return table

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
title = doc.add_heading('ICU Patient Deterioration Prediction', level=0)
for run in title.runs:
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
    run.font.size = Pt(22)

subtitle = doc.add_heading('Feature Documentation — Priority List', level=1)
for run in subtitle.runs:
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x66)
    run.font.size = Pt(14)

doc.add_paragraph('')
info = doc.add_paragraph()
info.add_run('Dataset: ').bold = True
info.add_run('eicu_hourly_labeled_6h_12h_24h.csv\n')
info.add_run('Source: ').bold = True
info.add_run('eICU Collaborative Research Database\n')
info.add_run('Total Records: ').bold = True
info.add_run('~94,049 hourly observations\n')
info.add_run('Total Columns: ').bold = True
info.add_run('48')

# ══════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════
add_heading_styled('Overview', 1)
doc.add_paragraph(
    'This document lists all features used in the ICU Patient Deterioration Prediction project, '
    'ordered by priority/importance based on the model\'s SHAP-derived feature weights and clinical '
    'relevance for predicting patient deterioration at 6-hour, 12-hour, and 24-hour horizons.'
)

# Category summary table
add_heading_styled('Feature Categories', 2)
make_table(
    ['Category', 'Count', 'Description'],
    [
        ['Vital Signs (Time-Series)', '7', 'Continuous physiological measurements recorded hourly'],
        ['Laboratory Values (Time-Series)', '9', 'Blood test results updated periodically'],
        ['Demographic & Administrative', '5', 'Static patient information'],
        ['Missingness Indicators', '16', 'Binary flags indicating imputed values'],
        ['Target Labels & Metadata', '8', 'Prediction targets and data splits (NOT used as inputs)'],
    ]
)

# ══════════════════════════════════════════════════════════════
# PRIORITY 1 — CRITICAL
# ══════════════════════════════════════════════════════════════
doc.add_page_break()
add_heading_styled('Priority 1 — Critical Predictors (Highest Importance)', 1)
p = doc.add_paragraph('These features have the ')
p.add_run('strongest influence').bold = True
p.add_run(' on deterioration prediction based on SHAP analysis.')

make_table(
    ['#', 'Feature', 'Type', 'Unit', 'Normal Range', 'Weight', 'Description'],
    [
        ['1', 'sao2', 'Vital Sign', '%', '95–100', '1.8',
         'Peripheral oxygen saturation (SpO₂). The single most important predictor — low SpO₂ strongly signals respiratory failure.'],
        ['2', 'Lactate', 'Lab Value', 'mmol/L', '0.5–2.0', '1.7',
         'Blood lactate level. Elevated lactate indicates tissue hypoperfusion, sepsis, or shock.'],
        ['3', 'systemicmean', 'Vital Sign', 'mmHg', '70–105', '1.6',
         'Mean Arterial Pressure (MAP). Critical indicator of organ perfusion. MAP < 65 mmHg signals septic shock.'],
        ['4', 'systemicsystolic', 'Vital Sign', 'mmHg', '90–140', '1.5',
         'Systolic Blood Pressure. Hypotension (< 90 mmHg) is a red flag for hemodynamic instability.'],
        ['5', 'Creatinine', 'Lab Value', 'mg/dL', '0.6–1.2', '1.5',
         'Serum creatinine. Rising creatinine indicates acute kidney injury (AKI).'],
    ],
    header_color='C0392B'
)

# ══════════════════════════════════════════════════════════════
# PRIORITY 2 — IMPORTANT
# ══════════════════════════════════════════════════════════════
add_heading_styled('Priority 2 — Important Predictors (High Importance)', 1)
doc.add_paragraph('These features significantly contribute to the model\'s predictions.')

make_table(
    ['#', 'Feature', 'Type', 'Unit', 'Normal Range', 'Weight', 'Description'],
    [
        ['6', 'heartrate', 'Vital Sign', 'bpm', '60–100', '1.4',
         'Heart rate. Tachycardia (> 100 bpm) or bradycardia (< 60 bpm) signals cardiovascular stress.'],
        ['7', 'respiration', 'Vital Sign', '/min', '12–20', '1.3',
         'Respiratory rate. Tachypnea (> 20/min) is one of the earliest signs of deterioration and a qSOFA criterion.'],
        ['8', 'WBC', 'Lab Value', '×10³/µL', '4.5–11.0', '1.2',
         'White Blood Cell count. Leukocytosis or leukopenia indicates infection/sepsis.'],
    ],
    header_color='E67E22'
)

# ══════════════════════════════════════════════════════════════
# PRIORITY 3 — MODERATE
# ══════════════════════════════════════════════════════════════
add_heading_styled('Priority 3 — Moderate Predictors', 1)
doc.add_paragraph('These features provide supporting information for prediction accuracy.')

make_table(
    ['#', 'Feature', 'Type', 'Unit', 'Normal Range', 'Weight', 'Description'],
    [
        ['9', 'temperature', 'Vital Sign', '°C', '36.1–37.5', '1.1',
         'Body temperature. Fever (> 38.3°C) or hypothermia (< 36°C) are SIRS criteria.'],
        ['10', 'BUN', 'Lab Value', 'mg/dL', '7–20', '1.1',
         'Blood Urea Nitrogen. Elevated BUN reflects renal dysfunction and dehydration.'],
        ['11', 'systemicdiastolic', 'Vital Sign', 'mmHg', '60–90', '1.0',
         'Diastolic Blood Pressure. Complements systolic BP in cardiovascular assessment.'],
        ['12', 'Potassium', 'Lab Value', 'mEq/L', '3.5–5.0', '1.0',
         'Serum potassium. Both hypo- and hyperkalemia cause dangerous cardiac arrhythmias.'],
        ['13', 'Platelets', 'Lab Value', '×10³/µL', '150–400', '1.0',
         'Platelet count. Thrombocytopenia is a component of DIC and sepsis coagulopathy.'],
    ],
    header_color='F1C40F'
)

# ══════════════════════════════════════════════════════════════
# PRIORITY 4 — SUPPORTING
# ══════════════════════════════════════════════════════════════
add_heading_styled('Priority 4 — Supporting Predictors (Lower Importance)', 1)
doc.add_paragraph('These features add supplementary context to the prediction model.')

make_table(
    ['#', 'Feature', 'Type', 'Unit', 'Normal Range', 'Weight', 'Description'],
    [
        ['14', 'Hemoglobin', 'Lab Value', 'g/dL', '12–17', '0.9',
         'Blood hemoglobin. Anemia (low Hgb) affects oxygen delivery and can worsen deterioration.'],
        ['15', 'Glucose', 'Lab Value', 'mg/dL', '70–140', '0.8',
         'Blood glucose. Both hypoglycemia and hyperglycemia are associated with poor ICU outcomes.'],
        ['16', 'Sodium', 'Lab Value', 'mEq/L', '136–145', '0.7',
         'Serum sodium. Dysnatremia affects neurological function and fluid balance.'],
    ],
    header_color='27AE60'
)

# ══════════════════════════════════════════════════════════════
# PRIORITY 5 — DEMOGRAPHIC
# ══════════════════════════════════════════════════════════════
doc.add_page_break()
add_heading_styled('Priority 5 — Demographic & Contextual Features', 1)
doc.add_paragraph('Static features providing patient background context.')

make_table(
    ['#', 'Feature', 'Type', 'Values/Range', 'Description'],
    [
        ['17', 'age (age_numeric)', 'Demographic', 'Numeric (years)',
         'Patient age. Older patients have higher baseline risk.'],
        ['18', 'gender (gender_encoded)', 'Demographic', 'Male / Female',
         'Patient biological sex. Encoded as binary for the model.'],
        ['19', 'ethnicity', 'Demographic', 'Caucasian, African American, etc.',
         'Patient ethnicity. Contextual demographic information.'],
        ['20', 'unitadmitsource', 'Administrative', 'ICU, ER, Floor, etc.',
         'Source of ICU admission — indicates severity at entry.'],
        ['21', 'unitstaytype', 'Administrative', 'admit, readmit, stepdown',
         'Type of ICU stay — readmissions may indicate higher risk.'],
    ],
    header_color='2980B9'
)

# ══════════════════════════════════════════════════════════════
# PRIORITY 6 — MISSINGNESS
# ══════════════════════════════════════════════════════════════
add_heading_styled('Priority 6 — Missingness Indicator Features', 1)
doc.add_paragraph(
    'Binary flags (0 or 1) indicating whether each corresponding feature value was imputed '
    '(missing in the original record). Missingness itself can be clinically informative — '
    'a test not ordered may indicate stability.'
)

missing_features = [
    'heartrate', 'systemicsystolic', 'systemicdiastolic', 'systemicmean',
    'respiration', 'sao2', 'temperature', 'WBC', 'Hemoglobin', 'Platelets',
    'Creatinine', 'BUN', 'Sodium', 'Potassium', 'Glucose', 'Lactate'
]
missing_rows = [[str(i+22), f'{f}_missing', f] for i, f in enumerate(missing_features)]
make_table(['#', 'Missingness Feature', 'Original Feature'], missing_rows, header_color='7F8C8D')

# ══════════════════════════════════════════════════════════════
# TARGETS (excluded)
# ══════════════════════════════════════════════════════════════
add_heading_styled('Target Variables (NOT Used as Input Features)', 1)
p = doc.add_paragraph('These columns are ')
p.add_run('prediction targets').bold = True
p.add_run(' and metadata — they are ')
p.add_run('excluded').bold = True
p.add_run(' from model input to prevent data leakage.')

make_table(
    ['Column', 'Description'],
    [
        ['label_6h', 'Binary — will the patient deteriorate within the next 6 hours?'],
        ['label_12h', 'Binary — will the patient deteriorate within the next 12 hours?'],
        ['label_24h', 'Binary — will the patient deteriorate within the next 24 hours?'],
        ['icu_death', 'Binary — did the patient die during the ICU stay?'],
        ['hours_to_event', 'Hours until deterioration event'],
        ['discharge_hour', 'Hour of ICU discharge'],
        ['unitdischargeoffset', 'Discharge offset in minutes'],
        ['split', 'Data split: train / val / test'],
    ],
    header_color='95A5A6'
)

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
add_heading_styled('Feature Engineering in Model Pipeline', 1)
doc.add_paragraph(
    'For classical ML models (Logistic Regression, Random Forest, XGBoost), '
    'the following summary features are derived from each 24-hour time-series window:'
)

make_table(
    ['Derived Feature', 'Description'],
    [
        ['Last Value', 'Most recent hourly reading for each feature'],
        ['Mean', 'Average value across the 24-hour observation window'],
        ['Standard Deviation', 'Variability/instability measure over 24 hours'],
        ['Linear Trend', 'Slope over time — captures worsening or improving trajectory'],
    ],
    header_color='8E44AD'
)

doc.add_paragraph('')
note = doc.add_paragraph()
note.add_run('Note: ').bold = True
note.add_run(
    'For deep learning models (LSTM, GRU), raw 24-hour sequences of all features '
    'are used directly as input without summary feature engineering.'
)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
add_heading_styled('Summary Statistics', 1)
make_table(
    ['Metric', 'Value'],
    [
        ['Total columns in dataset', '48'],
        ['Input features (used by model)', '37'],
        ['Excluded columns (targets + metadata)', '8'],
        ['Identifier columns', '3'],
        ['Vital sign features', '7'],
        ['Lab value features', '9'],
        ['Demographic features', '5'],
        ['Missingness indicators', '16'],
    ]
)

# Save
output_path = r'c:\Users\LAPTOP\Downloads\database preproccesssing\Feature_List_Priority.docx'
doc.save(output_path)
print(f'Done! Document saved to: {output_path}')
