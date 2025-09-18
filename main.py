from ultralytics import YOLOE
from ultralytics.utils.patches import torch_load
from ultralytics.models.yolo.yoloe import YOLOEPETrainer


model = YOLOE('yoloe-11s-seg.pt')
model = YOLOE('yoloe-11s.yaml')
state = torch_load('yoloe-11s-seg.pt')
model.load(state['model'])

head_index = len(model.model.model) - 1 # type: ignore
freeze = [str(i) for i in range(head_index)]
freeze.append(f"{head_index}.reprta")
freeze.append(f"{head_index}.savpe")

results = model.train(
    data='VOC.yaml',
    batch=16,
    epochs=15,
    close_mosaic=5,
    optimizer="AdamW",
    lr0=16e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    trainer=YOLOEPETrainer,
    device='0',
    pretrained='yoloe-11s-det.pt',
    val=True,
    freeze=freeze,
    project='experiments', name='baseline'
)
