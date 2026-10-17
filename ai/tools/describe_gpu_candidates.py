"""Optional local GPU descriptions, kept separate from detector scores and rules."""
import json
import os
from pathlib import Path

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def describe(video, candidates, model_config, maximum=8):
    import cv2
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    from prepare_gpu_dataset import frame_clip
    config=json.loads(Path(model_config).read_text(encoding='utf-8'))
    if config['model']!='Qwen/Qwen2.5-VL-7B-Instruct':
        raise ValueError('Expected the known Qwen2.5-VL-7B base model')
    if not 1<=maximum<=8:
        raise ValueError('Description limit must be 1..8')
    if not candidates:
        return candidates
    location=snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
    processor=AutoProcessor.from_pretrained(location,local_files_only=True,
                                             min_pixels=448*448,max_pixels=448*448)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(location,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,device_map={'':0},
        attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)).eval()
    capture=cv2.VideoCapture(str(video))
    try:
        fps=capture.get(cv2.CAP_PROP_FPS)
        count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps<=0 or count<=0:
            raise ValueError('Cannot decode video')
        duration=count/fps
    finally:
        capture.release()
    prompt=('Опиши по-русски в одном-двух коротких предложениях только видимые действия '
            'на этом коротком видео. Не называй людей и не читай персональные данные. '
            'Не утверждай наличие нарушения, намерения, принадлежность денег или применимость закона. '
            'Если действие или предмет неразборчивы, прямо укажи это. '
            'Если здесь нет кассовой сцены, опиши видимые объекты без выдумывания кассира. '
            'Не выдавай юридическое заключение. Без списков и без ссылок.')
    messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    result=[dict(item) for item in candidates]
    with torch.inference_mode():
        for event in result[:maximum]:
            center=event['timecode_sec']
            frames=frame_clip(video,center,duration,size=448,span=7)
            span=min(duration-.001,center+3.5)-max(0,center-3.5)
            batch=processor(text=[text],videos=[frames],fps=3/max(.001,span),
                            do_sample_frames=False,return_tensors='pt').to('cuda')
            output=model.generate(**batch,max_new_tokens=96,do_sample=False,use_cache=True)
            answer=processor.tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            if not answer:
                raise ValueError('Empty model description')
            event['observation_draft']=answer[:1600]
            del batch,output
    return result
