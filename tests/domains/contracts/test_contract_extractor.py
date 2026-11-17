from graphrag.domains.contracts import LICENSE_AGREEMENT_PATH, Contract, build_contract_extractor


class ParsingLLM:
    def __init__(self):
        self.calls = []

    def parse(self, messages, response_model, *, model, temperature=0.0):
        self.calls.append((messages, response_model, model))
        return "parsed"


def test_contract_extractor_sends_the_contract_prompt_and_model():
    llm = ParsingLLM()

    result = build_contract_extractor(llm, model="test-model").extract("an agreement")

    [(messages, response_model, model)] = llm.calls
    assert result == "parsed"
    assert response_model is Contract
    assert model == "test-model"
    assert "contract" in messages[0]["content"].lower()
    assert messages[1] == {"role": "user", "content": "an agreement"}


def test_license_agreement_is_bundled():
    assert "LICENSE" in LICENSE_AGREEMENT_PATH.read_text(encoding="utf-8").upper()
