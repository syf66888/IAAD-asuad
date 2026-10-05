"""Check NVIDIA runtime, model imports, and the real Java caption evaluator."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torchvision
from src.modeling.asuad_model import AsuadModel
from src.modeling.load_swin import get_swin_model
from src.modeling.load_bert import get_bert_model
from src.tasks.train import get_custom_args
from src.evalcap.runtime_preflight import validate_caption_metrics

if __name__ == '__main__':
    print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__,
                      'cuda_runtime': torch.version.cuda,
                      'cuda_available': torch.cuda.is_available(),
                      'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                      'model_imports': 'passed', 'caption_evaluator': validate_caption_metrics()}, indent=2))
