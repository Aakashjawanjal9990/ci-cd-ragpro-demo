"""
Text Cleaner, Legal Chunker & Contract Generator Module
Performs OCR cleanup, page-aware chunking, metadata extraction, and formats the JSON contract.
Supports both fast rule-based processing and LLM-assisted processing using Kedar's prompt contract.
Enhanced for both modern and historical (1950+ S.C.R.) Indian Supreme Court judgments.
"""

import re
import json
import logging
from typing import List, Tuple, Optional, Dict, Any

from models import (
    PageMetadata,
    ChunkMetadata,
    ChunkRecord,
    PageIngestionContract
)

logger = logging.getLogger(__name__)

# Common OCR typo corrections for Indian Supreme Court judgments (including historical prints)
OCR_REPLACEMENTS = [
    (r'\bC0urt\b', 'Court'),
    (r'\bC0URT\b', 'COURT'),
    (r'\bJudgm[e3]+nt\b', 'Judgment'),
    (r'\bJUDGM[E3]+NT\b', 'JUDGMENT'),
    (r'\bApp[e3]+llant\b', 'Appellant'),
    (r'\bAPP[E3]+LLANT\b', 'APPELLANT'),
    (r'\bResp[0o]+ndent\b', 'Respondent'),
    (r'\bRESP[0o]+NDENT\b', 'RESPONDENT'),
    (r'\bPetiti[0o]+ner\b', 'Petitioner'),
    (r'\bPETITI[0o]+NER\b', 'PETITIONER'),
    (r'\bJ\s+U\s+D\s+G\s+M\s+E\s+N\s+T\b', 'JUDGMENT'),
    (r'\bO\s+R\s+D\s+E\s+R\b', 'ORDER'),
    (r'\bv[e3]+rsus\b', 'versus'),
    (r'\bV[e3]+rsus\b', 'Versus'),
    (r'\bVERS[U0]+S\b', 'VERSUS'),
    (r'\bH[0o]+n\'?ble\b', "Hon'ble"),
    (r'\bJ\s+u\s+s\s+t\s+i\s+c\s+e\b', 'Justice'),
    (r'\bJ\s+U\s+S\s+T\s+I\s+C\s+E\b', 'JUSTICE'),
    (r'\bS\s+U\s+P\s+R\s+E\s+M\s+E\b', 'SUPREME'),
    (r'\bJncome-tax\b', 'Income-tax'),
    (r'\bGovl!rnment\b', 'Government'),
    (r'\b0entral\b', 'Central'),
    (r'\bExciieli\b', 'Excise'),
]


class LegalTextCleaner:
    """Cleans OCR artifacts and normalizes text without altering legal meaning."""

    @staticmethod
    def clean_text(raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text

        # 1. Fix line-break hyphenation (e.g. "repre- \nsentative" -> "representative")
        text = re.sub(r'(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)', r'\1\2', text)

        # 2. Strip recurring historical S.C.R. running page headers (including OCR typos like 'OOURT', 'SUPR!!:ME', 'COUR1'')
        text = re.sub(
            r'(?:^|\n)\s*(?:[-•,;~`\'\s]*\n)*\s*'
            r'(?:(?:S\.?C\.?R\.?|\[?\d{4}\]?|\d{1,4})\s*\n\s*)?'
            r'(?:SUPR[E!]+ME\s+[CO0]{1,2}UR[T1\']+\s+REPORTS?)\s*\n\s*'
            r'(?:(?:S\.?C\.?R\.?|\[?\d{4}\]?|\d{1,4})\s*\n)*',
            '\n',
            text,
            flags=re.IGNORECASE
        )

        # 3. Fix OCR substitutions
        for pattern, replacement in OCR_REPLACEMENTS:
            text = re.sub(pattern, replacement, text)

        # 4. Normalize carriage returns
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Replace 3+ consecutive newlines with 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Normalize intra-line spaces (multiple spaces/tabs to single space)
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        cleaned_text = '\n'.join(lines).strip()

        return cleaned_text


class LegalMetadataExtractor:
    """Extracts case metadata from Indian Supreme Court page text using robust pattern matching."""

    @staticmethod
    def extract_metadata(page_text: str) -> PageMetadata:
        metadata = PageMetadata()

        # Clean text for metadata extraction by stripping markdown artifacts if present
        plain_text = re.sub(r'[*#_]', '', page_text)

        # 1. Case Number pattern (Civil Appeal, Criminal Appeal, SLP, Writ Petition, etc.)
        case_no_match = re.search(
            r'(?:(?:CIVIL|CRIMINAL)\s+APPEAL\s+NO[S]?\.?|SPECIAL\s+LEAVE\s+PETITION\s*(?:\([A-Z]+\))?\s*NO[S]?\.?|SLP\s*(?:\([A-Z]+\))?\s*NO[S]?\.?|WRIT\s+PETITION\s*(?:\([A-Z]+\))?\s*NO[S]?\.?|REVIEW\s+PETITION\s*(?:\([A-Z]+\))?\s*NO[S]?\.?|CASE\s+NO[S]?\.?)\s*(?:OF|/)?\s*[\d\s/\-ofOF]+(?:\s+OF\s+\d{4})?',
            plain_text,
            re.IGNORECASE
        )
        if case_no_match:
            cand = ' '.join(case_no_match.group(0).split())
            if re.search(r'\d+', cand):
                metadata.case_number = cand

        if not metadata.case_number:
            # Check for specific Supreme Court Reports volume citation with page numbers: e.g., "[1950] S.C.R. 15"
            scr_match = re.search(r'\[?\d{4}\]?\s+S\.C\.R\.?\s+\d+', plain_text, re.IGNORECASE)
            if scr_match:
                cand = ' '.join(scr_match.group(0).split())
                if cand.upper() != "S.C.R." and re.search(r'\d+', cand):
                    metadata.case_number = cand

        # 2. Appellant and Respondent extraction
        # Try S.C.R. multi-line uppercase block first
        vs_pos = re.search(r'\n\s*(?:VERSUS|VS\.?|Vs\.|v\.|v)\s*\n', plain_text, re.IGNORECASE)
        if vs_pos:
            before_text = plain_text[:vs_pos.start()]
            after_text = plain_text[vs_pos.end():]

            # Get uppercase lines immediately preceding 'v.'
            lines_before = [l.strip() for l in before_text.split('\n') if l.strip()]
            app_lines = []
            for l in reversed(lines_before):
                # Stop if we hit headers, numbers, appeal numbers, or non-party delimiters
                if re.match(r'^(?:S\.C\.R\.|SUPREME|COURT|REPORTS|\[|\d+|JUDGMENT|ORDER|AGENT|CIVIL|CRIMINAL|APPEAL|SPECIAL|LEAVE|PETITION|WRIT|SLP|CASE|NO)', l, re.IGNORECASE):
                    break
                app_lines.insert(0, l)
                if len(app_lines) >= 3:
                    break

            if app_lines:
                app_cand = ' '.join(app_lines)
                app_cand = re.sub(r'\s*\.{2,}.*$', '', app_cand).strip().strip('.')
                app_cand = re.sub(r'\s*(?:Appellant|Petitioner).*$', '', app_cand, flags=re.IGNORECASE).strip()
                if len(app_cand) > 3 and not re.match(r'^(?:VERSUS|VS|S\.C\.R\.)$', app_cand, re.IGNORECASE):
                    metadata.appellant = app_cand

            # Get lines immediately following 'v.' up to end of party description
            lines_after = [l.strip() for l in after_text.split('\n') if l.strip()]
            resp_lines = []
            for l in lines_after:
                if re.match(r'^(?:UNION\s+OF\s+INDIA|INTERVENER|\[|\d+|DATED|CORAM|JUDGMENT|ORDER|HELD)', l, re.IGNORECASE):
                    break
                resp_lines.append(l)
                if l.endswith('.') or len(resp_lines) >= 3:
                    break

            if resp_lines:
                resp_cand = ' '.join(resp_lines)
                resp_cand = re.sub(r'\s*\.{2,}.*$', '', resp_cand).strip().strip('.')
                resp_cand = re.sub(r'\s*(?:Respondent|Defendant|Interveners?).*$', '', resp_cand, flags=re.IGNORECASE).strip()
                if len(resp_cand) > 3 and not re.match(r'^(?:VERSUS|VS|S\.C\.R\.)$', resp_cand, re.IGNORECASE):
                    metadata.respondent = resp_cand

        # Fallback to single-line format: e.g. "Ram Krishna v. Secretary, Municipal Committee"
        if not metadata.appellant or not metadata.respondent:
            single_vs = re.search(r'\n([A-Z\s,]{4,})\s+(?:v\.|vs\.|versus)\s+([A-Z\s,]{4,})(?:\.|\n)', plain_text, re.IGNORECASE)
            if single_vs:
                if not metadata.appellant:
                    cand = ' '.join(single_vs.group(1).split())
                    if not re.match(r'^(?:VERSUS|VS|S\.C\.R\.)$', cand, re.IGNORECASE):
                        metadata.appellant = cand
                if not metadata.respondent:
                    cand = ' '.join(single_vs.group(2).split())
                    if not re.match(r'^(?:VERSUS|VS|S\.C\.R\.)$', cand, re.IGNORECASE):
                        metadata.respondent = cand

        # 3. Date of Judgment (Clean newlines and OCR noise first)
        normalized_date_text = re.sub(r'\s+', ' ', plain_text)
        date_match = re.search(
            r'(?:DATED|Dated|DATED ON|DATED THIS|PRONOUNCED ON|DATE OF JUDGMENT)\s*[:\-]?\s*([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+[0-9]{4}|\d{1,2}[./-]\d{1,2}[./-]\d{4})',
            normalized_date_text,
            re.IGNORECASE
        )
        if date_match:
            metadata.date_of_judgment = date_match.group(1).strip()
        else:
            # Match dates specifically in 1950s SCR format (e.g. "1950 March 14" or "March 14, 1950")
            scr_date = re.search(
                r'\b(\d{4}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2})\b',
                normalized_date_text,
                re.IGNORECASE
            )
            if scr_date:
                metadata.date_of_judgment = scr_date.group(1).strip()
            else:
                alt_date = re.search(
                    r'\b([0-9]{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})\b',
                    normalized_date_text,
                    re.IGNORECASE
                )
                if alt_date:
                    metadata.date_of_judgment = alt_date.group(1).strip()

        # 4. Judges / Coram
        judges: List[str] = []

        # Check for bracketed Coram lists: e.g. "[SHRI HARILAL KANIA C.J., SAIYID FAZL ALI, ... JJ.]"
        coram_match = re.search(r'\[\s*(?:SHRI\s+|HON\'BLE\s+)?([A-Z\s.,]+(?:C\.J\.|JJ\.|J\.))\s*\]', plain_text, re.IGNORECASE)
        if coram_match:
            raw_coram = coram_match.group(1).replace('\n', ' ')
            parsed_judges = [
                re.sub(r'^(?:SHRI|HON\'BLE)\s+', '', j.strip(), flags=re.IGNORECASE).strip('., ')
                for j in re.split(r',|\band\b', raw_coram, flags=re.IGNORECASE)
            ]
            for pj in parsed_judges:
                clean_pj = ' '.join(pj.split())
                if len(clean_pj) > 3 and clean_pj not in judges:
                    judges.append(clean_pj)

        # Standard modern format judges search
        judge_matches = re.findall(
            r'(?:(?:Hon\'ble\s+)?(?:Mr\.|Ms\.|Mrs\.|Dr\.)?\s*Justice\s+[^\n,]+|(?:(?:Hon\'ble\s+)?(?:Dr\.|Mr\.|Ms\.|Mrs\.)\s+)?[A-Z][a-zA-Z\s.]*?(?:,\s*CJI|,\s*J\.))',
            plain_text
        )
        for jm in judge_matches:
            clean_j = ' '.join(jm.strip().split()).strip('.,')
            clean_j = re.sub(r'^(?:JUDGMENT|ORDER)\s+', '', clean_j, flags=re.IGNORECASE)
            if clean_j and clean_j not in judges and len(clean_j) > 5:
                judges.append(clean_j)

        if judges:
            metadata.judges = judges

        return metadata


class PageChunker:
    """Splits cleaned page text into logical, traceable chunks while preserving section headings."""

    HEADING_PATTERNS = [
        r'^(?:J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|JUDGMENT)$',
        r'^(?:O\s*R\s*D\s*E\s*R|ORDER)$',
        r'^(?:BACKGROUND|FACTS\s+AND\s+BACKGROUND|FACTS|ISSUES|SUBMISSIONS(?:\s+OF\s+COUNSEL)?|ANALYSIS(?:\s+AND\s+CONCLUSION)?|DISCUSSION|CONCLUSION|VERDICT|OPERATIVE ORDER)$',
        r'^(?:PART\s+[A-Z0-9]+|SECTION\s+[A-Z0-9]+|POINT\s+[A-Z0-9]+)$',
    ]

    @classmethod
    def is_heading(cls, text: str) -> bool:
        clean = re.sub(r'^[#\s*]+', '', text).rstrip('*').strip().upper()
        # Reject judge running headers from being treated as section headings
        if re.search(r'(?:C\.J\.|JJ\.|J\.)$', clean):
            return False
        for pat in cls.HEADING_PATTERNS:
            if re.match(pat, clean):
                return True
        return False

    @classmethod
    def get_clean_heading(cls, text: str) -> str:
        """Returns clean normalized heading title."""
        clean = re.sub(r'^[#\s*]+', '', text).rstrip('*').strip()
        return clean

    @classmethod
    def chunk_page(cls, document_id: str, page_number: int, cleaned_text: str) -> List[ChunkRecord]:
        """
        Splits page text into logical chunks:
        - Paragraphs separated by blank lines or distinct list items
        - Preserves section headings across following chunks
        - Assigns deterministic chunk IDs linked to document and original page
        """
        if not cleaned_text:
            return []

        # Step 1: Break text into paragraph blocks and isolate headings
        lines = cleaned_text.split('\n')
        blocks: List[str] = []
        curr_block: List[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                if curr_block:
                    blocks.append('\n'.join(curr_block))
                    curr_block = []
                continue

            if cls.is_heading(line_str):
                if curr_block:
                    blocks.append('\n'.join(curr_block))
                    curr_block = []
                blocks.append(line_str)
            else:
                curr_block.append(line_str)

        if curr_block:
            blocks.append('\n'.join(curr_block))

        # Step 2: Traverse blocks, update active heading, and create chunks
        chunks: List[ChunkRecord] = []
        current_heading: Optional[str] = None
        chunk_idx = 1

        for block in blocks:
            b_clean = block.strip()
            if not b_clean:
                continue

            # Check if this entire block is a heading
            if cls.is_heading(b_clean):
                current_heading = cls.get_clean_heading(b_clean)
                continue

            # Check for inline heading at the beginning (e.g. "JUDGMENT: ...")
            heading_split = re.match(
                r'^(?:[#\s*]*)(J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|JUDGMENT|O\s*R\s*D\s*E\s*R|ORDER)(?:[*#\s]*)\s*[:\-]\s*(.*)$',
                b_clean,
                re.IGNORECASE | re.DOTALL
            )
            if heading_split:
                current_heading = cls.get_clean_heading(heading_split.group(1))
                b_clean = heading_split.group(2).strip()

            # Sub-split blocks if there are distinct numbered paragraphs (e.g. "\n1. ... \n2. ...")
            sub_points = re.split(r'\n(?=\d+\.\s+|\[\d+\]\s+|\([a-z0-9]+\)\s+)', b_clean)
            for sp in sub_points:
                sp_clean = sp.strip()
                if not sp_clean:
                    continue

                # If a subpoint is itself a heading, update active heading
                if cls.is_heading(sp_clean):
                    current_heading = cls.get_clean_heading(sp_clean)
                    continue

                # Filter out pure noise / running headers from becoming standalone chunks (Issue H4)
                is_pure_noise = bool(re.match(
                    r'^(?:VERSUS|VS\.?|V\.|v|\d+|\d+\s*SUPREME\s*COURT\s*REPORTS|\[?\d{4}\]?\s*S\.C\.R\.?)$',
                    sp_clean,
                    re.IGNORECASE
                ))
                if is_pure_noise:
                    if chunks:
                        chunks[-1].content += f"\n{sp_clean}"
                    continue

                # Merge very short unstructured fragments (<80 chars) into previous chunk unless it's a distinct numbered point
                is_numbered_point = bool(re.match(r'^(?:\d+\.\s+|\[\d+\]\s+|\([a-z0-9]+\)\s+)', sp_clean))
                if not is_numbered_point and len(sp_clean) < 80 and chunks and (len(chunks[-1].content) + len(sp_clean) < 1400):
                    chunks[-1].content += f"\n{sp_clean}"
                    continue

                chunk_id = f"{document_id}_p{page_number}_c{chunk_idx}"
                chunk = ChunkRecord(
                    chunk_index=chunk_idx,
                    content=sp_clean,
                    chunk_metadata=ChunkMetadata(section_heading=current_heading),
                    chunk_id=chunk_id
                )
                chunks.append(chunk)
                chunk_idx += 1

        # Fallback: if no chunks were formed, treat whole cleaned_text as chunk 1
        if not chunks and cleaned_text:
            chunks.append(
                ChunkRecord(
                    chunk_index=1,
                    content=cleaned_text,
                    chunk_metadata=ChunkMetadata(section_heading=current_heading),
                    chunk_id=f"{document_id}_p{page_number}_c1"
                )
            )

        return chunks


class PageProcessor:
    """
    Coordinates cleaning, metadata extraction, and chunking to produce the PageIngestionContract.
    Can operate via rule-based fast mode or LLM-based mode using Kedar's exact prompt.
    """

    EXTRACTION_PROMPT_TEMPLATE = """You are an expert legal data extraction and processing agent specialized in Indian Supreme Court judgments. Your task is to process raw, potentially noisy PDF/OCR text from a single page and convert it into a clean, structured JSON contract.

### Strict Rules:
1. OCR Cleanup: Correct obvious scanning typos (e.g., 'C0urt' to 'Court'), but DO NOT alter the legal meaning, summarize, or rephrase the text.
2. Page-Aware Chunking: Split the cleaned text into logical chunks (e.g., paragraphs or distinct list items). 
3. Traceability: You must include the provided Document ID and Page Number exactly as given.
4. Output Format: Return ONLY a valid JSON object matching the exact structure below. Do not include markdown formatting or conversational text.

### Required JSON Structure:
{{
  "document_id": "String (Must match the provided Source Document ID)",
  "original_page_number": "Integer (Must match the provided Page Number)",
  "page_metadata": {{
    "case_number": "String or null (Extract if present on this page)",
    "appellant": "String or null (Extract if present on this page)",
    "respondent": "String or null (Extract if present on this page)",
    "date_of_judgment": "String or null (Extract if present on this page)",
    "judges": ["Array of Strings or null (Extract if present on this page)"]
  }},
  "chunks": [
    {{
      "chunk_index": "Integer (1, 2, 3... sequential order for this page)",
      "content": "String (The exact cleaned text of the chunk)",
      "chunk_metadata": {{
        "section_heading": "String or null (If the chunk falls under a specific heading)"
      }}
    }}
  ]
}}

### Input Data:
Source Document ID: {document_id}
Page Number: {page_number}
Raw Text to Process:
{raw_page_text}"""

    @classmethod
    def process_page_rule_based(cls, document_id: str, page_number: int, raw_text: str) -> PageIngestionContract:
        """Processes a single page using fast, deterministic rule-based algorithms."""
        cleaned_text = LegalTextCleaner.clean_text(raw_text)
        metadata = LegalMetadataExtractor.extract_metadata(cleaned_text)
        chunks = PageChunker.chunk_page(document_id, page_number, cleaned_text)

        return PageIngestionContract(
            document_id=document_id,
            original_page_number=page_number,
            page_metadata=metadata,
            chunks=chunks
        )

    @classmethod
    def format_llm_prompt(cls, document_id: str, page_number: int, raw_page_text: str) -> str:
        """Populates Kedar's exact prompt template with parameters."""
        return cls.EXTRACTION_PROMPT_TEMPLATE.format(
            document_id=document_id,
            page_number=page_number,
            raw_page_text=raw_page_text
        )

    @classmethod
    def parse_llm_json_response(cls, document_id: str, page_number: int, response_text: str) -> PageIngestionContract:
        """Parses and validates LLM response according to the contract, with fallback."""
        text = response_text.strip()
        # Strip potential markdown fences if returned
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        contract = PageIngestionContract(**data)
        # Assign deterministic chunk_ids if missing
        for c in contract.chunks:
            if not c.chunk_id:
                c.chunk_id = f"{document_id}_p{page_number}_c{c.chunk_index}"
        return contract
