from typing import Dict

VOCAB = [
    # Chiffres et point décimal
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.',
    # Booléens
    'true', 'false',
    # Registres
    'r0',  'r1',  'r2',  'r3',  'r4',  'r5',  'r6',  'r7',
    'r8',  'r9',  'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
    # Opérateurs arithmétiques
    '+', '-', '*', '/', '%',
    # Opérateurs logiques
    'and', 'or', 'not',
    # Opérateurs de comparaison
    '==', '!=', '<', '>', '<=', '>=',
    # Assignation et délimiteurs
    '=', '(', ')', ';', '\n',
    # Mots-clés
    'if', 'then', 'else', 'endif',
    'while', 'do', 'endwhile',
    'break', 'continue',
    'input', 'output',
]

TOKEN_TO_ID: Dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {idx: tok for idx, tok in enumerate(VOCAB)}

VOCAB_SIZE = len(VOCAB)  # 58