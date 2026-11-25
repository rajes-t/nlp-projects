# Resume Screening and Candidate Ranking System

## Project Architecture, Stage-by-Stage Workflow, and Technology Stack

**Version:** 1.0\
**Development environment:** Jupyter Notebook\
**Primary models:** `BAAI/bge-large-en-v1.5` and
`yashpwr/resume-ner-bert`\
**Output:** Pandas DataFrames displayed in the notebook

------------------------------------------------------------------------

## 1. Project objective

Build an NLP-based resume screening system that accepts one Job
Description (JD) and multiple DOCX resumes, identifies comparable
sections, measures semantic relevance, ranks candidates, and extracts
structured candidate information from the top five resumes.

The system is intended to support recruiter review. Similarity scores
are relevance indicators, not probabilities of job success and not
automatic hiring or rejection decisions.

### Scope and decisions

  Item                 Project decision
  -------------------- ----------------------------------------
  Development          Jupyter Notebook
  JD input             DOCX file path defined in the notebook
  Resume input         Folder path defined in the notebook
  Resume format        DOCX
  OCR                  Not required
  PDF processing       Not included in this version
  Interface            No Streamlit or web interface
  Embedding model      `BAAI/bge-large-en-v1.5`
  Similarity           Cosine similarity
  NER model            `yashpwr/resume-ner-bert`
  Contact extraction   Regex, with validation
  Final results        Pandas DataFrames
  File export          Not required

------------------------------------------------------------------------

## 2. Client-facing architecture diagram

``` mermaid
flowchart TD
    A["Input: One JD DOCX + Multiple Resume DOCX Files"] --> B["Document Extraction<br/>python-docx"]
    B --> C["Text Validation and Light Cleaning<br/>Python + Regex"]
    C --> D["Section Detection and Standardization"]

    D --> E1["JD Sections<br/>JD heading aliases"]
    D --> E2["Resume Sections<br/>Resume heading aliases"]

    E1 --> F["Section-aware Chunking<br/>Sentence-aware + Token-based"]
    E2 --> F

    F --> G["Embedding Generation<br/>BAAI/bge-large-en-v1.5"]
    G --> H["Section/Chunk Similarity<br/>Cosine Similarity"]
    H --> I["Weighted Candidate Score<br/>Skills 40% | Experience 35%<br/>Projects 15% | Other 10%"]
    I --> J["Rank All Candidates<br/>Select Top 5"]

    J --> K["Candidate Information Extraction<br/>yashpwr/resume-ner-bert + Regex"]
    K --> L["Combine Ranking and Candidate Details"]
    L --> M["Final Pandas DataFrames<br/>Candidate Ranking + Extracted Details"]
```

------------------------------------------------------------------------

## 3. Stage-by-stage implementation plan

### Stage 1 --- Define input paths and discover files

**Objective:** Point the notebook to one JD and a folder containing the
resumes.

**Tools** - Python variables - `pathlib` - `glob` or `Path.glob()`

**Implementation** 1. Define the JD DOCX file path. 2. Define the resume
folder path. 3. Find all `.docx` files in the folder. 4. Assign a stable
resume ID to each file, for example `RES_001`. 5. Validate that the JD
and resume files exist.

**Expected output** - JD file path - List of resume file paths - Resume
ID and filename mapping

No file-upload interface is required.

### Stage 2 --- Extract text from DOCX documents

**Objective:** Extract text from the JD and each resume.

**Tools** - `python-docx` (`from docx import Document`)

**Implementation** 1. Read paragraphs in document order. 2. Extract
non-empty paragraph text. 3. Extract table cell text as well, because
resumes often place skills and contact details in tables. 4. Preserve
paragraph boundaries and document order. 5. Retain the original
extracted text for debugging and traceability.

**Important limitation** - This version supports DOCX files with
selectable text. - OCR and scanned-document processing are intentionally
excluded. - DOCX files can contain tables, headers, footers, columns,
and text boxes. The extraction implementation should be tested against
the actual resume templates; ordinary paragraph and table extraction may
not capture every text box or complex layout.

**Expected output** - JD raw text - A mapping of each resume ID to its
raw extracted text

### Stage 3 --- Validate and lightly clean text

**Objective:** Normalize formatting without damaging information needed
for matching or entity extraction.

**Tools** - Python - `re` (Regex) - `unicodedata` if normalization is
needed

**Cleaning operations** 1. Normalize line endings and whitespace. 2.
Remove repeated blank lines. 3. Normalize bullet characters where
appropriate. 4. Remove clearly irrelevant formatting artifacts. 5.
Preserve numbers, dates, email addresses, phone numbers, skill names,
punctuation within technical terms, and section boundaries.

**Avoid** - Removing all stopwords. - Stemming or lemmatizing the entire
document. - Removing numbers or dates. - Aggressive punctuation removal.

Keep both the original text and cleaned text.

### Stage 4 --- Detect and standardize JD and resume sections

**Objective:** Identify document sections and map different heading
names to a common schema.

Use two separate heading-alias dictionaries: one for JDs and one for
resumes. Heading matching should be performed against normalized
paragraph/heading text, not against arbitrary occurrences of words
inside sentences.

#### 4A. Resume heading aliases

  -----------------------------------------------------------------------
  Standard section                    Example heading aliases
  ----------------------------------- -----------------------------------
  `summary`                           Summary, Professional Summary,
                                      Career Summary, Profile, About Me,
                                      Objective, Career Objective

  `skills`                            Skills, Technical Skills, Core
                                      Skills, Key Skills, Core
                                      Competencies, Technical Expertise

  `experience`                        Experience, Work Experience,
                                      Professional Experience, Employment
                                      History, Career History

  `projects`                          Projects, Academic Projects,
                                      Personal Projects, Key Projects

  `education`                         Education, Academic Qualifications,
                                      Educational Background,
                                      Qualifications

  `certifications`                    Certifications, Professional
                                      Certifications, Courses, Licenses
  -----------------------------------------------------------------------

#### 4B. JD heading aliases

  -----------------------------------------------------------------------
  Standard section                    Example heading aliases
  ----------------------------------- -----------------------------------
  `job_title`                         Job Title, Position, Role,
                                      Designation, Hiring For

  `summary`                           Job Summary, Role Overview,
                                      Position Overview, About the Role,
                                      Job Description

  `responsibilities`                  Responsibilities, Key
                                      Responsibilities, Duties, Roles and
                                      Responsibilities, What You'll Do

  `skills`                            Required Skills, Technical Skills,
                                      Core Skills, Must-Have Skills, Key
                                      Skills, Requirements

  `experience`                        Experience, Required Experience,
                                      Work Experience, Years of
                                      Experience, Experience Required

  `education`                         Education, Educational
                                      Qualifications, Academic
                                      Requirements, Qualifications

  `preferred`                         Preferred Qualifications, Good to
                                      Have, Nice to Have, Desired Skills,
                                      Additional Requirements

  `certifications`                    Certifications, Required
                                      Certifications, Professional
                                      Certifications
  -----------------------------------------------------------------------

#### Section detection rules

1.  Normalize heading case, whitespace, punctuation, and trailing
    colons.
2.  Match known aliases to standard section names.
3.  Treat the text following a detected heading as belonging to that
    section until the next recognized heading.
4.  Preserve unrecognized headings and their content under an `other` or
    `unclassified` section.
5.  Do not silently discard content if no heading is detected.
6.  If the same section appears more than once, append the content or
    preserve separate subsections under the same standard key.
7.  Review detection results for a sample of documents before
    proceeding.

**Fallback for documents without clear headings** - Preserve the full
text. - Use conservative rules or a later classification step to
identify likely content. - Do not guess a section when the evidence is
weak; mark it as unclassified for review.

**Expected output example**

``` python
{
    "RES_001": {
        "summary": "...",
        "skills": "...",
        "experience": "...",
        "projects": "...",
        "education": "...",
        "certifications": "...",
        "other": "..."
    }
}
```

The JD should be stored using its own standardized section dictionary.

### Stage 5 --- Section-aware chunking

**Objective:** Keep short sections intact and split long sections into
model-safe chunks without losing context.

**Why chunking is needed** - `BAAI/bge-large-en-v1.5` has a maximum
sequence length of 512 tokens. - Long resume or JD sections must not be
passed as one oversized input. - The tokenizer, rather than word count
or character count, must be used to measure token length.

#### Recommended initial chunking configuration

  -----------------------------------------------------------------------
  Parameter                                               Initial setting
  ------------------------------ ----------------------------------------
  BGE maximum sequence length                                  512 tokens

  Target chunk size                                            400 tokens

  Chunk overlap                                                 50 tokens

  Short sections                                        Keep as one chunk

  Medium sections                                 Sentence-aware grouping

  Long sections                      Token-aware splitting, preferably at
                                                      sentence boundaries
  -----------------------------------------------------------------------

These are starting values to validate, not universal optimal settings.
Confirm the model's effective maximum sequence length and special-token
behavior from the loaded tokenizer/model.

#### Hybrid chunking strategy

1.  **Short section:** If it fits within the safe token budget, retain
    it as one chunk.
2.  **Medium section:** Group complete sentences until the target token
    budget is reached.
3.  **Long section:** Split into chunks using the BGE tokenizer, trying
    to preserve sentence boundaries.
4.  **Overlap:** Carry a small amount of context from the previous chunk
    into the next chunk.
5.  **Metadata:** Store resume ID, standard section name, chunk index,
    and token count with every chunk.

Do not split every sentence into its own embedding by default. A chunk
should contain enough surrounding context to represent a useful piece of
information.

**Example chunk record**

``` python
{
    "resume_id": "RES_001",
    "section": "experience",
    "chunk_id": 1,
    "text": "...",
    "token_count": 382
}
```

Apply the same chunking principles to JD sections and resume sections.

### Stage 6 --- Generate embeddings

**Objective:** Convert each JD and resume chunk into a dense vector.

**Model** - `BAAI/bge-large-en-v1.5` - Hugging Face `transformers` or
Sentence Transformers

**Recommended implementation** - Load the model and tokenizer once. -
Use the model's tokenizer to prepare inputs. - Generate embeddings in
batches where practical. - Use the model's recommended pooling and
normalization approach. - Use inference mode to avoid storing training
gradients. - Store each vector with its document ID, section, and chunk
ID.

**Expected output** - One embedding for every JD chunk. - One embedding
for every resume chunk. - BGE-large-en-v1.5 produces 1024-dimensional
embeddings.

**Important** Follow the model card's recommended embedding procedure.
Do not assume that arbitrary pooling of token outputs is equivalent to
the model's intended sentence-embedding representation.

### Stage 7 --- Calculate section-wise and chunk-wise similarity

**Objective:** Compare relevant JD content with the corresponding resume
content.

**Tools** - `scikit-learn` (`cosine_similarity`) - NumPy

**Initial comparison plan**

  -----------------------------------------------------------------------
  JD section                          Resume section to compare
  ----------------------------------- -----------------------------------
  Skills / required skills            Skills; relevant evidence in
                                      experience and projects can be
                                      considered separately

  Experience                          Experience

  Responsibilities                    Experience and projects

  Education                           Education

  Certifications                      Certifications

  Summary                             Summary
  -----------------------------------------------------------------------

For each comparable section: 1. Compare JD chunk embeddings with resume
chunk embeddings. 2. Create a chunk-to-chunk cosine similarity matrix.
3. For each JD chunk, find its best matching resume chunk. 4. Aggregate
the best matches across JD chunks to obtain a section score.

**Initial aggregation strategy**

-   Calculate the maximum similarity for each JD chunk against the
    candidate's chunks in the corresponding section.
-   Average those per-JD-chunk maximum scores to produce the section
    score.
-   Retain the best-matching chunk pairs as evidence.

This avoids averaging every possible chunk pair, which can unfairly
lower a score when a resume contains additional unrelated content.

**Additional safeguards** - Avoid allowing one resume chunk to
repeatedly satisfy many unrelated JD requirements without review. -
Consider top-k averaging or one-to-one chunk matching as later
experiments. - If a required JD section or candidate section is missing,
record it as missing rather than treating it as a normal zero without an
explicit scoring policy. - Cosine similarity is a relevance signal, not
a probability or a verified skill claim.

### Stage 8 --- Calculate weighted candidate scores and rank candidates

**Objective:** Give the highest importance to skills, followed by
experience, then projects. Other sections receive lower priority.

#### Initial proposed weightage

  Matching category             Weight
  ------------------------- ----------
  Skills                       **40%**
  Experience                   **35%**
  Projects                     **15%**
  Other relevant sections      **10%**
  **Total**                   **100%**

The 10% "Other" category can include summary, education, and
certifications. These can be subdivided later if required.

#### Initial scoring formula

``` text
Final Score =
    (Skills Score     * 0.40) +
    (Experience Score * 0.35) +
    (Projects Score   * 0.15) +
    (Other Score      * 0.10)
```

The weights are configurable starting values and should be reviewed with
recruiters and tested on labelled examples.

**Missing-section policy** - Define in advance how missing sections are
treated. - Do not automatically penalize a candidate merely because a
resume uses a different heading. - If a category is genuinely absent,
either mark it as unavailable and redistribute weights among available
categories, or apply a clearly documented missing-evidence policy. -
Display missing evidence separately from low similarity.

**Additional requirement checks** - Check explicit minimum years of
experience using extracted dates/durations and transparent rules. -
Distinguish required skills from preferred skills. - Do not rely on
semantic similarity alone to verify numerical requirements.

**Expected ranking output**

  -----------------------------------------------------------------------------
  Resume ID     Skills   Experience   Projects      Other      Final       Rank
                                                               score 
  --------- ---------- ------------ ---------- ---------- ---------- ----------
  RES_001         0.91         0.88       0.85       0.80       0.88          1

  RES_002         0.87         0.82       0.79       0.84       0.84          2
  -----------------------------------------------------------------------------

*Scores above are illustrative only.*

Rank candidates by the final score and select the top five for detailed
information extraction. Keep the full ranking available for recruiter
review.

### Stage 9 --- Extract candidate information from the top five resumes

**Objective:** Extract structured candidate details after shortlisting.

**Primary NER model** - `yashpwr/resume-ner-bert` - Hugging Face
`transformers`

**Additional extraction** - Regex for email addresses and phone
numbers. - Custom rules for fields that the NER model does not reliably
identify.

**Potential fields** - Candidate name - Email - Phone number - Skills -
Company names - Job titles - Education - Experience-related entities -
Certifications

Before implementation, inspect the model's actual label set and model
card. Extract only fields supported by its labels; use Regex or custom
rules for other fields.

**Long-document handling** - Respect the NER model's own maximum
sequence length; it may differ from BGE's. - Tokenize and process long
resume sections in manageable windows. - Use overlap where appropriate
so entities near chunk boundaries are not lost. - Merge duplicate
entities across overlapping windows. - Validate email and phone
candidates before displaying them.

**Expected output** A structured record for each shortlisted resume,
linked to its resume ID.

### Stage 10 --- Combine results and display Pandas DataFrames

**Objective:** Present candidate rankings and extracted information
clearly in Jupyter Notebook.

**Tools** - Pandas - `pprint` for inspecting nested dictionaries -
Jupyter display

#### DataFrame A --- Candidate ranking

  -----------------------------------------------------------------------------
        Rank Resume ID     Skills   Experience   Projects      Other      Final
                            score        score      score      score      score
  ---------- --------- ---------- ------------ ---------- ---------- ----------
           1 RES_001         0.91         0.88       0.85       0.80       0.88

           2 RES_002         0.87         0.82       0.79       0.84       0.84
  -----------------------------------------------------------------------------

#### DataFrame B --- Shortlisted candidate details

  ---------------------------------------------------------------------------------
  Resume ID   Name        Email               Phone        Skills      Experience
  ----------- ----------- ------------------- ------------ ----------- ------------
  RES_001     Example     example@email.com   0000000000   Python, NLP 4 years
              Candidate                                                

  ---------------------------------------------------------------------------------

*All example values are illustrative.*

No CSV, Excel, JSON, or Pickle export is required for this version. The
results will be displayed directly in Pandas DataFrames.

------------------------------------------------------------------------

## 4. Technology stack by stage

  ----------------------------------------------------------------------------
  Stage                   Component               Tool / library
  ----------------------- ----------------------- ----------------------------
  1                       Path and file discovery `pathlib`, Python

  2                       DOCX extraction         `python-docx`

  3                       Text cleaning           `re`, Python

  4                       Section identification  Python dictionaries, Regex
                                                  and custom rules

  5                       Token-aware chunking    BGE tokenizer, Python

  6                       Embedding generation    `BAAI/bge-large-en-v1.5`,
                                                  Transformers / Sentence
                                                  Transformers, PyTorch

  7                       Similarity calculation  Scikit-learn, NumPy

  8                       Ranking and weighted    Pandas, Python
                          scoring                 

  9                       Resume NER              `yashpwr/resume-ner-bert`,
                                                  Transformers

  9                       Email and phone         Regex
                          extraction              

  10                      Results display         Pandas, Jupyter Notebook,
                                                  `pprint`
  ----------------------------------------------------------------------------

### Suggested Python dependencies

``` bash
pip install python-docx
pip install pandas
pip install numpy
pip install scikit-learn
pip install sentence-transformers
pip install transformers
pip install torch
pip install accelerate
```

The exact installation versions should be pinned after testing in the
chosen environment. A GPU is optional; inference can also run on CPU,
although it may be slower.

------------------------------------------------------------------------

## 5. Evaluation and quality checks

Before treating the ranking as useful, test it with manually reviewed
examples.

### Document processing checks

-   Did all DOCX files load?
-   Was text extracted from paragraphs and tables?
-   Were section headings detected correctly?
-   Was any content accidentally dropped or assigned to the wrong
    section?
-   Were long sections chunked without exceeding the model's token
    limit?

### Embedding and ranking checks

-   Are the most relevant resumes ranked near the top?
-   Do skill and experience matches influence ranking as intended?
-   Are missing sections handled consistently?
-   Are the matching chunks useful evidence for the score?

### NER checks

-   Is the candidate name correct?
-   Are email and phone values extracted accurately?
-   Are skills, companies, job titles, and education mapped to the
    correct fields?
-   Are duplicate entities caused by chunk overlap removed?

### Useful evaluation measures

-   Precision@5 and Recall@5 for shortlist quality.
-   NER precision, recall, and F1 for entity extraction.
-   Manual review of top-ranked candidates and their supporting
    evidence.

------------------------------------------------------------------------

## 6. Important design considerations

1.  **Similarity is not qualification verification.** A high cosine
    similarity score does not prove that a candidate has a required
    skill or the required number of years of experience.
2.  **Keep evidence.** Retain the JD chunk and resume chunk responsible
    for each strong match so a recruiter can inspect the result.
3.  **Avoid sensitive or irrelevant ranking features.** Do not rank
    candidates using age, gender, caste, religion, photographs, or other
    protected characteristics.
4.  **Keep personal data secure.** Limit access to resumes and extracted
    contact information, and define a retention and deletion policy.
5.  **Human review is required.** The system should assist recruiters,
    not make automatic hiring or rejection decisions.
6.  **Validate model suitability.** The embedding model and resume NER
    model must be evaluated on representative resumes before relying on
    their output.

------------------------------------------------------------------------

## 7. Final development sequence

Implement and validate one stage at a time:

1.  Define input paths and discover DOCX files.
2.  Extract JD and resume text.
3.  Validate extraction and perform light cleaning.
4.  Detect and standardize JD sections.
5.  Detect and standardize resume sections.
6.  Implement section-aware chunking for both document types.
7.  Load BGE and generate embeddings.
8.  Calculate section-wise chunk similarity.
9.  Calculate weighted candidate scores and rank all resumes.
10. Inspect and validate the top five.
11. Load the resume NER model and extract candidate information.
12. Combine ranking and entity information into Pandas DataFrames.

**Final deliverable:** A Jupyter Notebook-based resume screening
pipeline that displays a ranked candidate table, section-wise matching
scores, supporting matching evidence, and extracted details for the top
five candidates.
