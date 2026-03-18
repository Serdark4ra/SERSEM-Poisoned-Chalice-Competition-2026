import re
import numpy as np
import warnings
from typing import List, Optional, Dict

try:
    import enchant
    en_dict = enchant.Dict("en_US")
    ENCHANT_AVAILABLE = True
except Exception as e:
    warnings.warn(f"Failed to load pyenchant: {e}. Dictionary filtering disabled.")
    ENCHANT_AVAILABLE = False

# Attempt to load tree-sitter parsers for the 5 target languages
try:
    import tree_sitter
    import tree_sitter_python as tspython
    import tree_sitter_java as tsjava
    import tree_sitter_go as tsgo
    import tree_sitter_ruby as tsruby
    import tree_sitter_rust as tsrust

    PARSERS: Dict[str, tree_sitter.Parser] = {}
    
    def get_parser(lang_module):
        lang = tree_sitter.Language(lang_module.language())
        parser = tree_sitter.Parser(lang)
        return parser

    PARSERS['Python'] = get_parser(tspython)
    PARSERS['Java'] = get_parser(tsjava)
    PARSERS['Go'] = get_parser(tsgo)
    PARSERS['Ruby'] = get_parser(tsruby)
    PARSERS['Rust'] = get_parser(tsrust)
    AST_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to load tree-sitter parsers: {e}. Falling back completely to Regex.")
    AST_AVAILABLE = False


class ASTExtractor:
    """
    Parses code into byte-level weight masks across 5 languages,
    prioritizing Tree-Sitter AST with Dictionary checking, 
    and utilizing Regex as a safety net.
    """
    def __init__(self):
        self.W_STANDARD = 1.0
        self.W_IDENTIFIER_LONG = 3.0
        self.W_STRING = 5.0
        self.W_COMMENT = 10.0
        self.W_MULTILINGUAL = 10.0
        
        # Regex patterns 
        self.regex_comment_fb = re.compile(r'(//[^\n]*|#[^\n]*|/\*.*?\*/)', re.DOTALL)
        self.regex_string_fb = re.compile(r'(".*?"|\'.*?\')', re.DOTALL)
        self.regex_identifier_fb = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]{3,}\b')

    def split_identifier(self, ident: str) -> List[str]:
        words = []
        for p in ident.split('_'):
            subparts = re.findall(r'[A-Za-z][a-z0-9]*|[A-Z]+(?=[A-Z][a-z0-9]|\b)', p)
            if subparts:
                words.extend(subparts)
            elif p:
                words.append(p)
        return words

    def extract_weights(self, text: str, language: Optional[str] = None) -> np.ndarray:
        char_weights = np.full(len(text), self.W_STANDARD, dtype=float)
        byte_text = text.encode('utf-8')

        ast_success = False
        if AST_AVAILABLE and language in PARSERS:
            try:
                parser = PARSERS[language]
                tree = parser.parse(byte_text)
                
                def traverse_and_weight(node):
                    node_type = node.type
                    
                    char_start = len(byte_text[:node.start_byte].decode('utf-8', errors='replace'))
                    char_end = len(byte_text[:node.end_byte].decode('utf-8', errors='replace'))
                    
                    if 'comment' in node_type:
                        char_weights[char_start:char_end] = self.W_COMMENT
                    elif 'string' in node_type:
                        if char_end - char_start > 15:
                            char_weights[char_start:char_end] = self.W_STRING
                    elif 'identifier' in node_type:
                        ident_str = text[char_start:char_end]
                        is_multilingual = False
                        
                        if ENCHANT_AVAILABLE:
                            words = self.split_identifier(ident_str)
                            # Only flag as multilingual if a significant portion of words are unknown
                            # to avoid penalizing strictly valid programmatic camelCase/snake_case
                            unknown_count = sum(1 for w in words if len(w) >= 4 and not en_dict.check(w))
                            if unknown_count > 0 and len(words) > 0 and unknown_count / len(words) >= 0.5:
                                is_multilingual = True
                                
                        mask = char_weights[char_start:char_end] == self.W_STANDARD
                        if is_multilingual:
                            char_weights[char_start:char_end][mask] = self.W_MULTILINGUAL
                        elif char_end - char_start > 10:
                            char_weights[char_start:char_end][mask] = self.W_IDENTIFIER_LONG
                    
                    for child in node.children:
                        traverse_and_weight(child)

                traverse_and_weight(tree.root_node)
                ast_success = True
            except Exception as e:
                warnings.warn(f"AST Parsing failed for language {language}: {e}. Falling back to Regex.")

        # Fallback 
        if not ast_success:
            for match in self.regex_comment_fb.finditer(text):
                char_weights[match.start():match.end()] = self.W_COMMENT
            for match in self.regex_string_fb.finditer(text):
                if match.end() - match.start() > 15:
                    char_weights[match.start():match.end()] = self.W_STRING
            for match in self.regex_identifier_fb.finditer(text):
                if match.end() - match.start() > 10:
                    mask = char_weights[match.start():match.end()] == self.W_STANDARD
                    char_weights[match.start():match.end()][mask] = self.W_IDENTIFIER_LONG

        return char_weights
