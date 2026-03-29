from src.language.lexer import Lexer
from src.language.parser import Parser
from src.language.parser.ASTNodes import *

SOURCE = """
5
4.69
true
false

r0
r1
r2
r3
r4
r5
r6
r7
r8
r9
r10
r11
r12
r13
r14
r15

-7
+4.5
not true

-r0

1 + 2
47 - 58 * 6
(47 - 58) * 6
r0 + 7 * (4 + r1 / 5 % 7)
78 > 47
78 >= 47
78 < 47
78 <= 47
78 == 47
78 != 47
12 == 21 and 21 < 12
12 == 21 or 21 < 12

r9 = 42


if r0 > 0 then endif

if r0 > 0 then
    r1
endif

if r0 > 0 then
    r1
else
    r2
endif


while r3 do
    r3 = false
endwhile

while r3 do
    if r0 > 0 then break endif
    r0 = r0 + 1
endwhile

while r3 do
    if r3 then continue endif
endwhile

input r0
input r1
output r0 + r1
"""

class ParserTests:

    @staticmethod
    def _parse(source: str) -> ProgramNode:
        """Méthode utilitaire pour parser un bout de code rapidement."""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    # ─── 1. Tests des types de base ──────────────────────────────────────────

    @staticmethod
    def parse_numbers():
        ast = ParserTests._parse("5\n4.69")
        assert len(ast.statements) == 2
        
        stmt1, stmt2 = ast.statements
        
        assert isinstance(stmt1, NumberNode)
        assert stmt1.value == 5
        assert stmt1.type == NumberType.INTEGER

        assert isinstance(stmt2, NumberNode)
        assert stmt2.value == 4.69
        assert stmt2.type == NumberType.FLOAT

    @staticmethod
    def parse_booleans():
        ast = ParserTests._parse("true\nfalse")
        assert len(ast.statements) == 2
        
        stmt1, stmt2 = ast.statements
        assert isinstance(stmt1, BooleanNode)
        assert stmt1.value is True
        
        assert isinstance(stmt2, BooleanNode)
        assert stmt2.value is False

    @staticmethod
    def parse_registers():
        ast = ParserTests._parse("r0\nr15")
        assert len(ast.statements) == 2
        
        assert isinstance(ast.statements[0], RegisterNode)
        assert ast.statements[0].value == "r0"
        
        assert isinstance(ast.statements[1], RegisterNode)
        assert ast.statements[1].value == "r15"

    # ─── 2. Tests des Opérateurs ─────────────────────────────────────────────

    @staticmethod
    def parse_unary_operators():
        ast = ParserTests._parse("-7\nnot true")
        assert len(ast.statements) == 2
        
        stmt1, stmt2 = ast.statements
        
        assert isinstance(stmt1, UnaryOpNode)
        assert stmt1.operator == UnaryOperatorType.NEGATIVE
        assert isinstance(stmt1.operand, NumberNode)
        assert stmt1.operand.value == 7

        assert isinstance(stmt2, UnaryOpNode)
        assert stmt2.operator == UnaryOperatorType.NOT
        assert isinstance(stmt2.operand, BooleanNode)
        assert stmt2.operand.value is True

    @staticmethod
    def parse_binary_operators_and_precedence():
        # Test: 47 - 58 * 6
        # La multiplication doit être prioritaire, donc l'arbre est:
        # BinaryOp(-, 47, BinaryOp(*, 58, 6))
        ast = ParserTests._parse("47 - 58 * 6")
        stmt = ast.statements[0]
        
        assert isinstance(stmt, BinaryOpNode)
        assert stmt.operator == BinaryOperatorType.SUB
        assert isinstance(stmt.left, NumberNode)
        assert stmt.left.value == 47
        
        # Le nœud de droite doit être la multiplication
        right_node = stmt.right
        assert isinstance(right_node, BinaryOpNode)
        assert right_node.operator == BinaryOperatorType.MUL
        assert right_node.left.value == 58
        assert right_node.right.value == 6

    @staticmethod
    def parse_parentheses():
        # Test: (47 - 58) * 6
        # Les parenthèses forcent la soustraction en premier
        ast = ParserTests._parse("(47 - 58) * 6")
        stmt = ast.statements[0]
        
        assert isinstance(stmt, BinaryOpNode)
        assert stmt.operator == BinaryOperatorType.MUL
        assert isinstance(stmt.left, BinaryOpNode)
        assert stmt.left.operator == BinaryOperatorType.SUB
        assert stmt.right.value == 6

    # ─── 3. Tests des Assignations ───────────────────────────────────────────

    @staticmethod
    def parse_assignment():
        ast = ParserTests._parse("r9 = 42")
        stmt = ast.statements[0]
        
        assert isinstance(stmt, AssignmentNode)
        assert isinstance(stmt.target, RegisterNode)
        assert stmt.target.value == "r9"
        
        assert isinstance(stmt.value, NumberNode)
        assert stmt.value.value == 42

    # ─── 4. Tests de Contrôle de Flux ────────────────────────────────────────

    @staticmethod
    def parse_if_else():
        code = """
        if r0 > 0 then
            r1
        else
            r2
        endif
        """
        ast = ParserTests._parse(code)
        stmt = ast.statements[0]
        
        assert isinstance(stmt, IfNode)
        
        # Vérification de la condition (r0 > 0)
        assert isinstance(stmt.condition, BinaryOpNode)
        assert stmt.condition.operator == BinaryOperatorType.GT
        
        # Vérification du THEN
        assert len(stmt.then_block) == 1
        assert isinstance(stmt.then_block[0], RegisterNode)
        assert stmt.then_block[0].value == "r1"
        
        # Vérification du ELSE
        assert stmt.else_block is not None
        assert len(stmt.else_block) == 1
        assert isinstance(stmt.else_block[0], RegisterNode)
        assert stmt.else_block[0].value == "r2"

    @staticmethod
    def parse_while():
        code = """
        while r3 do
            break
            continue
        endwhile
        """
        ast = ParserTests._parse(code)
        stmt = ast.statements[0]
        
        assert isinstance(stmt, WhileNode)
        assert isinstance(stmt.condition, RegisterNode)
        assert stmt.condition.value == "r3"
        
        assert len(stmt.body) == 2
        assert isinstance(stmt.body[0], BreakNode)
        assert isinstance(stmt.body[1], ContinueNode)

    # ─── 5. Tests I/O ────────────────────────────────────────────────────────

    @staticmethod
    def parse_io():
        ast = ParserTests._parse("input r0\noutput r0 + 1")
        assert len(ast.statements) == 2
        
        stmt1, stmt2 = ast.statements
        
        assert isinstance(stmt1, InputNode)
        assert isinstance(stmt1.target, RegisterNode)
        assert stmt1.target.value == "r0"
        
        assert isinstance(stmt2, OutputNode)
        assert isinstance(stmt2.value, BinaryOpNode)
        assert stmt2.value.operator == BinaryOperatorType.ADD

    # ─── 6. Test Global ──────────────────────────────────────────────────────

    @staticmethod
    def parse_full_source():
        ast = ParserTests._parse(SOURCE)
        
        assert isinstance(ast, ProgramNode)
        assert len(ast.statements) > 10, "Le parser aurait dû trouver plusieurs dizaines d'instructions"