"""Offline experimental text-to-evidence adapter. Never produces a legal verdict."""
import argparse
import json
import os
from pathlib import Path
import sys

for key,value in {'USE_TF':'0','USE_FLAX':'0','HF_HUB_OFFLINE':'1',
                   'HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false'}.items():
    os.environ.setdefault(key,value)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rat_ai.evidence import CONDITIONS, assess, parse_facts
from train_rule_reasoning import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--rule-id',choices=CONDITIONS,required=True)
    parser.add_argument('--observation-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',action='store_true',help='Compare the same input without the new adapter')
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    config = json.loads((args.run/'config.json').read_text(encoding='utf-8'))
    if config['model']!='Qwen/Qwen3-VL-4B-Instruct' or config['dataset']['data_kind']!='fictional_public_rule_reasoning':
        raise ValueError('Not a rule-reasoning run')
    location = snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(location,local_files_only=True,trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        location,local_files_only=True,trust_remote_code=False,use_safetensors=True,
        device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True))
    if not args.base:
        model = PeftModel.from_pretrained(model,args.run/'adapter',local_files_only=True,is_trainable=False)
    model.eval()
    batch = encode(tokenizer,{'rule_id':args.rule_id,
        'observation':args.observation_file.read_text(encoding='utf-8')})
    if batch['input_ids'].shape[1] > 2048:
        raise ValueError('Observation exceeds tested context limit')
    batch = {k:v.to('cuda') for k,v in batch.items()}
    with torch.inference_mode():
        generated = model.generate(**batch,max_new_tokens=64,do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(generated[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    # Invalid output is an explicit error, not a successful all-unknown or clean result.
    result = assess(args.rule_id,parse_facts(text))
    result.update({'model':config['model'],'revision':config['revision'],
                   'adapter':'base' if args.base else 'public-rule-text-reasoning-v1',
                   'input_origin':'unverified_text', 'experimental':True})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    os.chmod(args.output,0o600)
    print(json.dumps({'written':True,'experimental':True,'legal_verdict':False}))


if __name__ == '__main__':
    main()
