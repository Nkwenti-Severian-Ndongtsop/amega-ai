import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
import tokenize
import io

@dataclass
class CodeChunk:
    content: str
    start_line: int
    end_line: int
    type: str  # 'function', 'class', 'block', or 'fragment'
    name: Optional[str] = None
    language: Optional[str] = None

class CodeChunker:
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 50, overlap: int = 20):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        
        # Language-specific patterns for improved chunking
        self.language_patterns = {
            'python': {
                'function_def': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'class_def': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]',
                'comment': r'#.*$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
            },
            'javascript': {
                'function_def': r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\(|([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?:async\s*)?\(',
                'class_def': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*{',
                'comment': r'\/\/.*$|\/\*[\s\S]*?\*\/',
            }
        }

    def detect_language(self, file_path: Union[str, Path]) -> str:
        """Detect the programming language based on file extension."""
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
        }
        return extensions.get(Path(file_path).suffix.lower(), 'unknown')

    def chunk_python_ast(self, code: str) -> List[CodeChunk]:
        """Chunk Python code using AST parsing for more accurate results."""
        chunks = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(CodeChunk(
                        content=ast.get_source_segment(code, node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        type='function',
                        name=node.name,
                        language='python'
                    ))
                elif isinstance(node, ast.ClassDef):
                    chunks.append(CodeChunk(
                        content=ast.get_source_segment(code, node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        type='class',
                        name=node.name,
                        language='python'
                    ))
        except SyntaxError:
            # Fallback to regex-based chunking if AST parsing fails
            return self.chunk_by_regex(code, 'python')
        return chunks

    def chunk_by_regex(self, code: str, language: str) -> List[CodeChunk]:
        """Chunk code using regex patterns when AST parsing is not available."""
        chunks = []
        lines = code.split('\n')
        patterns = self.language_patterns.get(language, {})
        
        current_chunk = []
        current_type = None
        current_name = None
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            for chunk_type, pattern in patterns.items():
                if match := re.search(pattern, line):
                    if current_chunk:
                        chunks.append(CodeChunk(
                            content='\n'.join(current_chunk),
                            start_line=start_line,
                            end_line=i-1,
                            type=current_type or 'block',
                            name=current_name,
                            language=language
                        ))
                        current_chunk = []
                    
                    current_type = chunk_type
                    current_name = match.group(1) if match.groups() else None
                    start_line = i
            
            current_chunk.append(line)
            
        if current_chunk:
            chunks.append(CodeChunk(
                content='\n'.join(current_chunk),
                start_line=start_line,
                end_line=len(lines),
                type=current_type or 'block',
                name=current_name,
                language=language
            ))
            
        return chunks

    def ensure_chunk_size(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Ensure chunks meet size constraints and add overlap where needed."""
        result = []
        for chunk in chunks:
            if len(chunk.content) <= self.max_chunk_size:
                result.append(chunk)
                continue
                
            # Split large chunks while preserving meaningful boundaries
            lines = chunk.content.split('\n')
            current_chunk = []
            current_size = 0
            start_line = chunk.start_line
            
            for i, line in enumerate(lines):
                line_size = len(line)
                if current_size + line_size > self.max_chunk_size and current_chunk:
                    # Add overlap from previous chunk if possible
                    overlap_start = max(0, len(current_chunk) - self.overlap)
                    overlap_lines = current_chunk[overlap_start:]
                    
                    result.append(CodeChunk(
                        content='\n'.join(current_chunk),
                        start_line=start_line,
                        end_line=start_line + len(current_chunk) - 1,
                        type='fragment',
                        language=chunk.language
                    ))
                    
                    current_chunk = overlap_lines + [line]
                    current_size = sum(len(l) for l in current_chunk)
                    start_line = start_line + len(current_chunk) - len(overlap_lines)
                else:
                    current_chunk.append(line)
                    current_size += line_size
            
            if current_chunk:
                result.append(CodeChunk(
                    content='\n'.join(current_chunk),
                    start_line=start_line,
                    end_line=start_line + len(current_chunk) - 1,
                    type='fragment',
                    language=chunk.language
                ))
                
        return result

    def chunk_code(self, code: str, file_path: Union[str, Path]) -> List[CodeChunk]:
        """Main method to chunk code with language-specific handling."""
        language = self.detect_language(file_path)
        
        if language == 'python':
            chunks = self.chunk_python_ast(code)
        else:
            chunks = self.chunk_by_regex(code, language)
            
        return self.ensure_chunk_size(chunks)

def create_embeddings(chunks: List[CodeChunk]):
    """
    Placeholder for future embedding generation functionality.
    This will be implemented separately to handle the actual embedding creation.
    """
    pass
