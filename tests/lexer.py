from src.language.lexer import Token, TokenType, Lexer

SOURCE = """
12
12.3
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
+
-
*
/
%
and
or
not
==
!=
<
>
<=
>=
=
(
)
;
if
then
else
endif
while
do
endwhile
break
continue
input
output
"""


class LexerTests:

    @staticmethod
    def token():
        token = Token(
            type=TokenType.REGISTER,
            value="r0",
            line=1,
            column=3
        )

        assert token.position == (1, 3), "Erreur : Token.position"

    @staticmethod
    def tokenize_method():
        lexer = Lexer(SOURCE)
        tokens = lexer.tokenize()
        
        filtered_tokens =[t for t in tokens if t.type != TokenType.NEWLINE]

        expected_tokens =[
            (TokenType.INTEGER, "12"),
            (TokenType.FLOAT, "12.3"),
            (TokenType.BOOLEAN, "true"),
            (TokenType.BOOLEAN, "false"),
            (TokenType.REGISTER, "r0"),
            (TokenType.REGISTER, "r1"),
            (TokenType.REGISTER, "r2"),
            (TokenType.REGISTER, "r3"),
            (TokenType.REGISTER, "r4"),
            (TokenType.REGISTER, "r5"),
            (TokenType.REGISTER, "r6"),
            (TokenType.REGISTER, "r7"),
            (TokenType.REGISTER, "r8"),
            (TokenType.REGISTER, "r9"),
            (TokenType.REGISTER, "r10"),
            (TokenType.REGISTER, "r11"),
            (TokenType.REGISTER, "r12"),
            (TokenType.REGISTER, "r13"),
            (TokenType.REGISTER, "r14"),
            (TokenType.REGISTER, "r15"),
            (TokenType.ADD, "+"),
            (TokenType.SUB, "-"),
            (TokenType.MUL, "*"),
            (TokenType.DIV, "/"),
            (TokenType.MOD, "%"),
            (TokenType.AND, "and"),
            (TokenType.OR, "or"),
            (TokenType.NOT, "not"),
            (TokenType.EQ, "=="),
            (TokenType.NEQ, "!="),
            (TokenType.LT, "<"),
            (TokenType.GT, ">"),
            (TokenType.LTE, "<="),
            (TokenType.GTE, ">="),
            (TokenType.ASSIGN, "="),
            (TokenType.LPAREN, "("),
            (TokenType.RPAREN, ")"),
            (TokenType.SEMICOLON, ";"),
            (TokenType.IF, "if"),
            (TokenType.THEN, "then"),
            (TokenType.ELSE, "else"),
            (TokenType.ENDIF, "endif"),
            (TokenType.WHILE, "while"),
            (TokenType.DO, "do"),
            (TokenType.ENDWHILE, "endwhile"),
            (TokenType.BREAK, "break"),
            (TokenType.CONTINUE, "continue"),
            (TokenType.INPUT, "input"),
            (TokenType.OUTPUT, "output"),
            (TokenType.EOF, "")
        ]

        assert len(filtered_tokens) == len(expected_tokens), \
            f"Erreur : Nombre de tokens | attendu {len(expected_tokens)} tokens, obtenu {len(filtered_tokens)}"

        for i, (token, (expected_type, expected_val)) in enumerate(zip(filtered_tokens, expected_tokens)):
            assert token.type == expected_type, \
                f"Erreur : Token {i} (Valeur '{token.value}') | Type attendu {expected_type}, obtenu {token.type}"