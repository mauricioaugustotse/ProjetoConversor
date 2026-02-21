import tempfile
import unittest
from pathlib import Path

import notion_trf1_canonical_agent as agent


class CanonicalAgentUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config_dir = Path(self._tmp.name) / "canon_config"
        self.rules = agent.CanonicalRules(self.config_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_relator_normalization_removes_prefix_keeps_convocado(self) -> None:
        got = self.rules.normalize_relator("des. federal Morais da Rocha (convocado)")
        self.assertEqual(got, "Morais da Rocha (convocado)")

    def test_informativo_numeric_validation(self) -> None:
        self.assertTrue(bool(agent.NUMERIC_INFORMATIVO_RE.fullmatch("742")))
        self.assertFalse(bool(agent.NUMERIC_INFORMATIVO_RE.fullmatch("unanimidade")))

    def test_classe_alias_mapping(self) -> None:
        self.assertEqual(self.rules.normalize_classe("ReeNeec"), "ReeNec")
        self.assertEqual(self.rules.normalize_classe("CJur"), "Cjur")
        self.assertEqual(self.rules.normalize_classe("MsCrim"), "MSCrim")

    def test_classe_new_pdf_siglas_supported(self) -> None:
        self.assertEqual(self.rules.normalize_classe("ReSE"), "ReSE")
        self.assertEqual(self.rules.normalize_classe("ED"), "ED")
        self.assertEqual(self.rules.normalize_classe("AgInt"), "AgInt")

    def test_subramo_cardinality_enforcement(self) -> None:
        allowed = {"Processo Penal", "Direito Penal Comum", "Direito Administrativo"}
        self.assertTrue(
            agent.subramo_is_valid_two(
                ["Processo Penal", "Direito Penal Comum"],
                self.rules,
                allowed,
            )
        )
        self.assertFalse(agent.subramo_is_valid_two(["Processo Penal"], self.rules, allowed))
        self.assertFalse(
            agent.subramo_is_valid_two(
                ["Processo Penal", "Direito Penal Comum", "Direito Administrativo"],
                self.rules,
                allowed,
            )
        )


if __name__ == "__main__":
    unittest.main()
