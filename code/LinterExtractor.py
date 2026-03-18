import re
import tempfile
import subprocess
from typing import List, Tuple

class LinterExtractor:
    """
    Extracts formatting anomalies (spacing, indentation, etc.) 
    using offline native linters or robust generic fallbacks.
    Returns a list of (line_num, col_num) tuples.
    """
    def __init__(self):
        # Universal regex to catch basic formatting chaos if natively unsupported
        self.erratic_spacing = re.compile(r'([^ ]  +[^ ])')  
        self.mixed_indent = re.compile(r'^(\t+ +| +\t+)', re.MULTILINE)
        
    def get_flake8_errors(self, text: str) -> List[Tuple[int, int]]:
        """Runs flake8 on python text."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
            f.write(text)
            f.flush()
            try:
                res = subprocess.run(['flake8', f.name], capture_output=True, text=True)
                errors = []
                for line in res.stdout.splitlines():
                    m = re.match(r'^.+?:(\d+):(\d+): (.+)', line)
                    if m:
                        errors.append((int(m.group(1)), int(m.group(2))))
                return errors
            except Exception:
                return []
                
    def get_generic_formatting_errors(self, text: str) -> List[Tuple[int, int]]:
        """A generic language-agnostic detector for severe formatting chaos."""
        errors = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line_num = i + 1
            # Check mixed indentation
            if self.mixed_indent.match(line):
                m = self.mixed_indent.search(line)
                errors.append((line_num, m.start() + 1))
            
            for m in self.erratic_spacing.finditer(line):
                if '//' not in line[:m.start()] and '#' not in line[:m.start()]:
                    errors.append((line_num, m.start() + 2))
                    
        return errors

    def extract_errors(self, text: str, language: str) -> List[Tuple[int, int]]:
        if language == 'Python':
            # Priority to Flake8 for absolute precision
            flake_errs = self.get_flake8_errors(text)
            if flake_errs: return flake_errs
            
        # Fallback to Generic formatting anomaly detection 
        return self.get_generic_formatting_errors(text)
