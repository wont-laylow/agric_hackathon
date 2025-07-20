import os
import re
import json
from pathlib import Path

class RAGDataIngestion:
    def __init__(self, base_dir: str, output_path: str):
        self.base_dir = base_dir
        self.output_path = output_path

    # Function to extract raw text from a markdown file
    def _extract_text(self, md_path: Path) -> str:
        with open(md_path, "r", encoding="utf-8") as file:
            return file.read()

    # Function to format markdown content into structured JSON
    def _structure_entry(self, crop: str, disease: str, content: str) -> dict:
        def _extract_section(text: str, header: str) -> str:
            pattern = rf"### {header}\n+(.*?)(?=\n### |\Z)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        return {
            "crop": crop,
            "disease": disease,
            "symptoms": _extract_section(content, "Symptoms"),
            "cause": _extract_section(content, "Cause"),
            "management": _extract_section(content, "Management"),
        }

    # Walk through the knowledge base directory and process markdown files
    def ingest_knowledge_base(self) -> list:
        data = []

        for crop_folder in Path(self.base_dir).iterdir():
            if crop_folder.is_dir():
                crop_name = crop_folder.name

                for md_file in crop_folder.glob("*.md"):
                    disease_name = md_file.stem
                    raw_content = self._extract_text(md_file)
                    entry = self._structure_entry(crop_name, disease_name, raw_content)
                    data.append(entry)

        return data

    # Save data to a JSON file
    @staticmethod
    def save_to_json(data: list, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
