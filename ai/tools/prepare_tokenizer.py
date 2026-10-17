"""Convert the already downloaded tokenizer to OpenVINO IR, entirely offline."""
import argparse
import os
from pathlib import Path
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("ai/local/models/qwen3-vl-2b-int4"))
    args = parser.parse_args()
    scratch = args.model.resolve() / ".conversion-tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model.resolve(), local_files_only=True, trust_remote_code=False)
    encoder, decoder = convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(encoder, str((args.model / "openvino_tokenizer.xml").resolve()))
    ov.save_model(decoder, str((args.model / "openvino_detokenizer.xml").resolve()))
    print("Offline tokenizer conversion completed")


if __name__ == "__main__":
    main()
