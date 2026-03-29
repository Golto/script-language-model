from typing import List, Tuple

from src.language.lexer import Lexer
from src.language.parser import Parser
from src.language.evaluator import Evaluator
from src.language.evaluator.environment import Environment, ValidRegisterType


class EvaluatorTests:

    @staticmethod
    def _evaluate(
        source: str, 
        env: Environment = None, 
        inputs: List[ValidRegisterType] = None
    ) -> Tuple[ValidRegisterType, Environment, List[ValidRegisterType]]:
        """
        Méthode utilitaire qui parse et évalue le code.
        Prend des entrées optionnelles (pour input) et retourne le dernier résultat, 
        l'environnement final, et la liste des sorties (output).
        """
        # Parsing
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        # Configuration de l'environnement et des I/O
        env = env or Environment()
        
        # Gestion des entrées simulées (mocks)
        input_queue = inputs.copy() if inputs else []
        def mock_input(register_name: str):
            if not input_queue:
                raise RuntimeError(f"Plus assez d'entrées disponibles pour {register_name}")
            return input_queue.pop(0)

        # Gestion des sorties simulées (mocks)
        outputs = []
        def mock_output(val: ValidRegisterType):
            outputs.append(val)

        # Évaluation
        evaluator = Evaluator(env=env, input_fn=mock_input, output_fn=mock_output)
        last_result = evaluator.visit(ast)
        
        return last_result, env, outputs

    # ─── 1. Tests des Expressions et Opérations ──────────────────────────────

    @staticmethod
    def eval_math_operations():
        source = """
        r0 = 10 + 5 * 2
        r1 = (10 + 5) * 2
        r2 = 10 / 2 - 1
        """
        _, env, _ = EvaluatorTests._evaluate(source)
        state = env.snapshot()
        
        assert state.get("r0") == 20
        assert state.get("r1") == 30
        assert state.get("r2") == 4

    @staticmethod
    def eval_logic_and_comparisons():
        source = """
        r0 = 10 > 5
        r1 = 10 == 10 and 5 < 2
        r2 = not (5 >= 10)
        """
        _, env, _ = EvaluatorTests._evaluate(source)
        state = env.snapshot()
        
        assert state.get("r0") is True
        assert state.get("r1") is False
        assert state.get("r2") is True

    # ─── 2. Tests de l'Environnement (Registres) ─────────────────────────────

    @staticmethod
    def eval_variables():
        env = Environment()
        env.set("r1", 15)
        
        source = """
        r2 = r1 + 5
        r1 = 100
        """
        _, final_env, _ = EvaluatorTests._evaluate(source, env=env)
        state = final_env.snapshot()
        
        assert state.get("r2") == 20
        assert state.get("r1") == 100

    # ─── 3. Tests du Contrôle de Flux (If / While) ───────────────────────────

    @staticmethod
    def eval_if_else():
        source = """
        r0 = 10
        if r0 > 5 then
            r1 = 1
        else
            r1 = 2
        endif
        
        if r0 == 0 then
            r2 = 1
        else
            r2 = 2
        endif
        """
        _, env, _ = EvaluatorTests._evaluate(source)
        state = env.snapshot()
        
        assert state.get("r1") == 1
        assert state.get("r2") == 2

    @staticmethod
    def eval_while_loop():
        source = """
        r0 = 3
        r1 = 0
        while r0 > 0 do
            r1 = r1 + 10
            r0 = r0 - 1
        endwhile
        """
        _, env, _ = EvaluatorTests._evaluate(source)
        state = env.snapshot()
        
        assert state.get("r0") == 0
        assert state.get("r1") == 30

    @staticmethod
    def eval_break_continue():
        source = """
        r0 = 0
        r1 = 0
        while r0 < 5 do
            r0 = r0 + 1
            if r0 == 2 then continue endif
            if r0 == 4 then break endif
            r1 = r1 + r0
        endwhile
        """
        _, env, _ = EvaluatorTests._evaluate(source)
        state = env.snapshot()
        
        assert state.get("r0") == 4
        assert state.get("r1") == 4

    # ─── 4. Tests des Entrées / Sorties (I/O) ────────────────────────────────

    @staticmethod
    def eval_io():
        source = """
        input r1
        input r2
        output r1 + r2
        output r1 > r2
        """
        _, _, outputs = EvaluatorTests._evaluate(source, inputs=[10, 5])
        
        assert len(outputs) == 2
        assert outputs[0] == 15
        assert outputs[1] is True

    # ─── 5. Test Global ──────────────────────────────────────────────────────

    @staticmethod
    def eval_full_script():
        source = """
        input r1
        input r2

        r1 = r1 / r2
        r3 = r1 < r2

        if r3 then 
            output r1 
        endif
        output r1 + r2
        output r3
        """
        _, env, outputs = EvaluatorTests._evaluate(source, inputs=[10, 2])
        
        state = env.snapshot()
        assert state.get("r1") == 5
        assert state.get("r2") == 2
        assert state.get("r3") is False

        assert outputs == [7, False]