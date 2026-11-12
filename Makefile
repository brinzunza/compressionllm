.PHONY: setup data test smoke experiments char-experiments report
setup:
	scripts/setup_vm.sh
data:
	scripts/get_data.sh
test:
	python3 -m pytest -q
smoke:
	cllm run configs/smoke_raw.yaml configs/smoke_rule.yaml configs/smoke.yaml
	cllm report --runs-dir runs_smoke
experiments:
	cllm run configs/exp_raw.yaml configs/exp_rule_vowel_blind.yaml configs/exp_rule_vowel_verified.yaml configs/exp_learned.yaml
	cllm report
char-experiments:
	cllm run configs/exp_char_raw.yaml configs/exp_char_unique.yaml configs/exp_char_rule_vowel_verified.yaml configs/exp_char_learned.yaml
	cllm report
report:
	cllm report
