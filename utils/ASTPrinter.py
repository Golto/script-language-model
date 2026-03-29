from typing import Any
import re


class ASTPrettyPrinter:
    """Formatte les AST de manière lisible avec indentation"""
    
    def __init__(self, indent_size: int = 4, show_position: bool = False, show_private: bool = False):
        self.indent_size = indent_size
        self.show_position = show_position
        self.show_private = show_private
    
    def format(self, node: Any) -> str:
        """Point d'entrée principal pour formatter un node"""
        return self._format_recursive(str(node), 0)
    
    def _format_recursive(self, text: str, depth: int) -> str:
        """Formatte récursivement en développant les parenthèses et crochets"""
        result = []
        i = 0
        current_line = ""
        indent = " " * (depth * self.indent_size)
        
        while i < len(text):
            char = text[i]
            
            # Ouvrir une parenthèse ou un crochet
            if char in '([':
                # Ajoute le caractère d'ouverture
                current_line += char
                result.append(indent + current_line)
                current_line = ""
                
                # Trouve le contenu entre les délimiteurs
                closing = ')' if char == '(' else ']'
                content, end_pos = self._extract_balanced(text, i + 1, closing)
                
                # Formatte le contenu
                formatted_content = self._format_content(content, depth + 1)
                result.extend(formatted_content)
                
                # Ajoute le caractère de fermeture
                result.append(indent + closing)
                
                i = end_pos + 1
                
            # Fermeture inattendue (ne devrait pas arriver)
            elif char in ')]':
                current_line += char
                i += 1
                
            # Caractère normal
            else:
                current_line += char
                i += 1
        
        # Ajoute la dernière ligne si elle existe
        if current_line.strip():
            result.append(indent + current_line)
        
        return '\n'.join(result)
    
    def _extract_balanced(self, text: str, start: int, closing_char: str) -> tuple[str, int]:
        """Extrait le contenu entre délimiteurs en gérant l'imbrication"""
        opening_char = '(' if closing_char == ')' else '['
        depth = 1
        i = start
        content = ""
        
        while i < len(text) and depth > 0:
            char = text[i]
            if char == opening_char:
                depth += 1
                content += char
            elif char == closing_char:
                depth -= 1
                if depth > 0:
                    content += char
            else:
                content += char
            i += 1
        
        return content, i - 1
    
    def _should_filter_argument(self, arg: str) -> bool:
        """Détermine si un argument doit être filtré"""
        arg_stripped = arg.strip()
        
        # Filtre position= si show_position est False
        if not self.show_position and arg_stripped.startswith('position='):
            return True
        
        # Filtre les arguments commençant par _ si show_private est False
        if not self.show_private and re.match(r'^_\w+\s*=', arg_stripped):
            return True
        
        return False
    
    def _format_content(self, content: str, depth: int) -> list[str]:
        """Formatte le contenu d'une parenthèse/crochet en séparant par virgules"""
        if not content.strip():
            return []
        
        # Sépare par virgules en tenant compte de l'imbrication
        parts = self._split_by_comma(content)
        
        # Filtre les arguments selon les paramètres
        filtered_parts = []
        for part in parts:
            if not self._should_filter_argument(part):
                filtered_parts.append(part)
        
        # Si tous les arguments ont été filtrés, retourne une liste vide
        if not filtered_parts:
            return []
        
        result = []
        indent = " " * (depth * self.indent_size)
        
        for i, part in enumerate(filtered_parts):
            part = part.strip()
            if not part:
                continue
            
            # Ajoute une virgule sauf pour le dernier élément
            suffix = "," if i < len(filtered_parts) - 1 else ""
            
            # Si la partie contient des parenthèses/crochets, formatte récursivement
            if '(' in part or '[' in part:
                formatted_part = self._format_recursive(part, depth)
                # Ajoute la virgule à la dernière ligne
                lines = formatted_part.split('\n')
                if lines:
                    lines[-1] += suffix
                result.extend(lines)
            else:
                result.append(indent + part + suffix)
        
        return result
    
    def _split_by_comma(self, text: str) -> list[str]:
        """Sépare par virgules en ignorant celles dans les parenthèses/crochets"""
        parts = []
        current = ""
        depth_paren = 0
        depth_bracket = 0
        
        for char in text:
            if char == '(':
                depth_paren += 1
                current += char
            elif char == ')':
                depth_paren -= 1
                current += char
            elif char == '[':
                depth_bracket += 1
                current += char
            elif char == ']':
                depth_bracket -= 1
                current += char
            elif char == ',' and depth_paren == 0 and depth_bracket == 0:
                parts.append(current)
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts


# Fonction helper pour usage simple
def format_ast(node: Any, indent_size: int = 4, show_position: bool = False, show_private: bool = False) -> str:
    """Formatte un node AST de manière lisible"""
    printer = ASTPrettyPrinter(indent_size, show_position, show_private)
    return printer.format(node)

# === TESTS ===
if __name__ == "__main__":
    # Test 1: Dictionnaire simple
    test1 = "DictionaryNode(pairs=[(StringNode(value='a'), NumberNode(value=1, type=NumberType.INTEGER)), (StringNode(value='b'), NumberNode(value=2, type=NumberType.INTEGER))])"
    
    # Test 2: Expression binaire simple
    test2 = "BinaryOpNode(left=NumberNode(value=1, type=NumberType.INTEGER), operator=BinaryOperatorType.PLUS, right=NumberNode(value=2, type=NumberType.INTEGER))"
    
    # Test 3: Expression complexe
    test3 = "BinaryOpNode(left=IdentifierNode(name='index'), operator=BinaryOperatorType.PLUS, right=BinaryOpNode(left=NumberNode(value=7, type=NumberType.INTEGER), operator=BinaryOperatorType.MUL, right=BinaryOpNode(left=BinaryOpNode(left=NumberNode(value=4, type=NumberType.INTEGER), operator=BinaryOperatorType.PLUS, right=BinaryOpNode(left=BinaryOpNode(left=IdentifierNode(name='divisor'), operator=BinaryOperatorType.DIV, right=NumberNode(value=5, type=NumberType.INTEGER)), operator=BinaryOperatorType.MOD, right=NumberNode(value=7, type=NumberType.INTEGER))), operator=BinaryOperatorType.POWER, right=NumberNode(value=2, type=NumberType.INTEGER))))"
    
    print("=" * 80)
    print("TEST 1: Dictionnaire simple")
    print("=" * 80)
    print(format_ast(test1))
    
    print("\n" + "=" * 80)
    print("TEST 2: Expression binaire simple")
    print("=" * 80)
    print(format_ast(test2))
    
    print("\n" + "=" * 80)
    print("TEST 3: Expression complexe imbriquée")
    print("=" * 80)
    print(format_ast(test3))
    
    print("\n" + "=" * 80)
    print("TEST 4: Avec indentation de 2")
    print("=" * 80)
    print(format_ast(test2, indent_size=2))