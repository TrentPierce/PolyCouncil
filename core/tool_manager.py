"""
Tool and File Management for PolyCouncil
Handles web search detection, visual model detection, and file parsing.
"""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import aiohttp

# Try to import file parsing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class FileParser:
    """Handles parsing of various file types and extracting text content."""
    
    @staticmethod
    def parse_file(file_path: Path) -> Optional[str]:
        """
        Parse a file and extract text content.
        Supports: TXT, PDF, DOCX
        
        Returns:
            Extracted text content or None if parsing fails
        """
        if not file_path.exists():
            return None
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                return file_path.read_text(encoding='utf-8', errors='ignore')
            
            elif suffix == '.pdf' and PDF_AVAILABLE:
                return FileParser._parse_pdf(file_path)
            
            elif suffix in ['.docx', '.doc'] and DOCX_AVAILABLE:
                return FileParser._parse_docx(file_path)
            
            else:
                # Try to read as text as fallback
                try:
                    return file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    return None
                    
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None
    
    @staticmethod
    def _parse_pdf(file_path: Path) -> Optional[str]:
        """Parse PDF file using PyPDF2."""
        if not PDF_AVAILABLE:
            return None
        try:
            text_parts = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text())
            return '\n\n'.join(text_parts)
        except Exception as e:
            print(f"PDF parsing error: {e}")
            return None
    
    @staticmethod
    def _parse_docx(file_path: Path) -> Optional[str]:
        """Parse DOCX file using python-docx."""
        if not DOCX_AVAILABLE:
            return None
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs]
            return '\n\n'.join(paragraphs)
        except Exception as e:
            print(f"DOCX parsing error: {e}")
            return None
    
    @staticmethod
    def format_context_block(parsed_text: str, filename: str = "document") -> str:
        """
        Format parsed text as a context block for injection into prompts.
        
        Args:
            parsed_text: The extracted text content
            filename: Name of the source file
            
        Returns:
            Formatted context block string
        """
        if not parsed_text or not parsed_text.strip():
            return ""
        
        return f"""=== CONTEXT FROM FILE: {filename} ===

{parsed_text.strip()}

=== END CONTEXT ===

"""


class ModelCapabilityDetector:
    """Detects model capabilities from OpenAI-compatible model metadata."""

    @staticmethod
    async def fetch_models_data(
        session: aiohttp.ClientSession,
        base_url: str,
        api_key: str = "",
        model_path: str = "v1/models",
    ) -> dict:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        models_url = f"{base_url.rstrip('/')}/{model_path.lstrip('/')}"
        async with session.get(models_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return {}
            return await resp.json()

    @staticmethod
    def _find_model_entry(models_data: dict, model: str) -> Optional[dict]:
        for item in models_data.get("data", []):
            if item.get("id") == model or item.get("name") == model:
                return item
        return None

    @staticmethod
    def detect_web_search_from_data(models_data: dict, model: str) -> bool:
        item = ModelCapabilityDetector._find_model_entry(models_data, model)
        if not item:
            return False
        capabilities = item.get("capabilities", {})
        if isinstance(capabilities, dict):
            if capabilities.get("web_search") or capabilities.get("tools"):
                return True
        model_name = (item.get("id") or item.get("name") or "").lower()
        if "search" in model_name or "tool" in model_name:
            return True
        model_info = item.get("info", {})
        if isinstance(model_info, dict):
            if model_info.get("supports_tools") or model_info.get("supports_functions"):
                return True
        return False

    @staticmethod
    def detect_visual_from_data(models_data: dict, model: str) -> bool:
        item = ModelCapabilityDetector._find_model_entry(models_data, model)
        if not item:
            return False
        capabilities = item.get("capabilities", {})
        if isinstance(capabilities, dict):
            if capabilities.get("vision") or capabilities.get("visual") or capabilities.get("multimodal"):
                return True
        model_name = (item.get("id") or item.get("name") or "").lower()
        if any(keyword in model_name for keyword in ["vision", "visual", "multimodal", "clip", "llava", "vl"]):
            return True
        model_info = item.get("info", {})
        if isinstance(model_info, dict):
            if model_info.get("supports_vision") or model_info.get("supports_images"):
                return True
        return False

    @staticmethod
    async def detect_web_search(session: aiohttp.ClientSession, base_url: str, model: str) -> bool:
        try:
            data = await ModelCapabilityDetector.fetch_models_data(session, base_url)
            return ModelCapabilityDetector.detect_web_search_from_data(data, model)
        except Exception as e:
            print(f"Error detecting web search for {model}: {e}")
            return False
    
    @staticmethod
    async def detect_visual(session: aiohttp.ClientSession, base_url: str, model: str) -> bool:
        try:
            data = await ModelCapabilityDetector.fetch_models_data(session, base_url)
            return ModelCapabilityDetector.detect_visual_from_data(data, model)
        except Exception as e:
            print(f"Error detecting visual capabilities for {model}: {e}")
            return False
    
    @staticmethod
    def encode_image(image_path: Path) -> Optional[str]:
        """
        Encode an image file to base64 for API transmission.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string or None if encoding fails
        """
        try:
            if not image_path.exists():
                return None
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
                encoded = base64.b64encode(image_data).decode('utf-8')
                
                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(str(image_path))
                if not mime_type:
                    # Default based on extension
                    suffix = image_path.suffix.lower()
                    mime_map = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }
                    mime_type = mime_map.get(suffix, 'image/jpeg')
                
                return f"data:{mime_type};base64,{encoded}"
                
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

