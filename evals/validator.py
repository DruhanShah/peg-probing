import torch
from data import PEG, get_dataloader


class Validator:
    def __init__(self, cfg, device, probe=None):
        self.cfg = cfg
        self.device = device
        self.dt = torch.bfloat16 if cfg.train.bf16 else torch.float32
        self.probe = probe

    def _calculate_binary_accuracy(self, predictions, targets):
        success = (predictions == targets).float()

        if success.dim() > 1:
            success = success.mean(dim=1)
        return success.mean().item()

    def _get_autocast_context(self):
        return torch.amp.autocast(device_type=self.device.split(":")[0],
                                  dtype=self.dt)

    def validate_model(self, model):
        if self.cfg.model.type == "recognizer":
            return self._binary_validation(model)
        elif self.cfg.model.type == "generator":
            return self._grammaticality_validation(model)

    def validate_probe(self, model, probe):
        return self._probe_validation(model, probe)

    def _binary_validation(self, model):
        dataloader = get_dataloader(
            self.cfg.lang, self.cfg.model_type,
            self.cfg.eval,
            self.cfg.work_dir, self.cfg.seed,
            kind="PEG", quiet=True
        )

        results = {"loss": [], "accuracy": []}
        debug_results = {"outputs": []}

        with torch.no_grad():
            for i, _in in enumerate(dataloader):
                inputs = _in["inputs"].to(self.device)
                outputs = _in["outputs"].to(self.device)

                with self._get_autocast_context(self.device):
                    # Forward pass through the model
                    _out = model(inputs, outputs,
                                 return_type=["logits", "loss"])
                    logits = _out["logits"].to(self.device)
                    loss = _out["loss"].mean().item()

                    # Get predictions and accuracy
                    pred = logits > 0
                    acc = self._calculate_binary_accuracy(pred, outputs)

                results["loss"].append(loss)
                results["accuracy"].append(acc)

        return results, debug_results

    def _grammaticality_validation(self, model):
        grammar = PEG(self.cfg.lang, max_length=self.cfg.eval.max_len)

        results = {"grammaticality": []}
        debug_results = {"outputs": []}

        with torch.no_grad():
            template = ""
            inputs = [
                torch.tensor(grammar.tokenize_string(template),
                             dtype=torch.long)[:-1].unsqueeze(0)
                for _ in range(self.cfg.eval.num_samples)
            ]

            for _in in inputs:
                with self._get_autocast_context():
                    N, T = self.cfg.eval.max_len, self.cfg.eval.temperature
                    _out = model.generate(_in, num_samples=N,
                                          temperature=T).tolist()
                    accuracy = 0
                    for pred in _out:
                        output = grammar.detokenize_string(pred, clean=True)
                        grammaticality = grammar.grammar_check(output)
                        accuracy += 1 if grammaticality else 0
                        debug_results["outputs"].append(output)
                    accuracy /= len(_out)
                    results["grammaticality"].append(accuracy)

        return results, debug_results

    def _probe_validation(self, model, probe):
        dataloader = get_dataloader(
            self.cfg.lang, self.cfg.probe_type,
            self.cfg.eval,
            self.cfg.work_dir, self.cfg.seed,
            kind="PS", quiet=True
        )

        results = {"loss": [], "accuracy": []}
        debug_results = {"outputs": []}

        with torch.no_grad():
            for _in in dataloader:
                inputs = _in["inputs"].to(self.device)
                outputs = _in["outputs"].to(self.device)

                with self._get_autocast_context():
                    # Forward pass through the model
                    _out = model(inputs, return_type=["cache"])
                    block0_out = _out["cache"]["block_0.mlp"]

                    # Forward pass through the probe
                    probe_out = probe(block0_out, outputs,
                                      return_type=["logits", "loss"])
                    logits = probe_out["logits"].to(self.device)
                    loss = probe_out["loss"].mean().item()

                    # Get vector of prediction successes
                    pred = logits > 0
                    acc = self._calculate_binary_accuracy(pred, outputs)

                results["loss"].append(loss)
                results["accuracy"].append(acc)

        return results, debug_results
