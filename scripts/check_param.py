import torch
import torchvision.models as models
from ultralytics import YOLO
from thop import profile, clever_format


def count_resnet_params_flops(arch_name='resnet18', input_size=320):
    model = getattr(models, arch_name)(weights=None)
    model.eval()
    
    dummy_input = torch.randn(1, 3, input_size, input_size)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.2f")
    
    print(f"\n{'='*50}")
    print(f"ResNet: {arch_name}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"{'='*50}")
    print(f"Parameters: {params_str} ({params:,.0f})")
    print(f"GFLOPs: {macs / 1e9:.2f} ({macs_str} MACs)")
    
    return params, macs


def count_yolo_params_flops(model_name='yolo11s.pt', input_size=640):
    model = YOLO(model_name)
    
    params = sum(p.numel() for p in model.model.parameters())
    
    dummy_input = torch.randn(1, 3, input_size, input_size)
    model.model.eval()
    macs, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
    
    params_str = clever_format([params], "%.2f")[0]
    
    print(f"\n{'='*50}")
    print(f"YOLO: {model_name}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"{'='*50}")
    print(f"Parameters: {params_str} ({params:,.0f})")
    print(f"GFLOPs: {macs / 1e9:.2f}")
    
    return params, macs


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FDAL Model Parameter & GFLOPs Analysis")
    print("="*60)
    
    yolo_params, yolo_macs = count_yolo_params_flops('yolo11s.pt', input_size=640)
    
    resnet_params, resnet_macs = count_resnet_params_flops('resnet18', input_size=320)
    
    print(f"\n{'='*60}")
    print("TOTAL (FDAL = YOLO + ResNet Supporter)")
    print(f"{'='*60}")
    total_params = yolo_params + resnet_params
    total_macs = yolo_macs + resnet_macs
    print(f"Total Parameters: {total_params / 1e6:.2f}M ({total_params:,.0f})")
    print(f"Total GFLOPs: {total_macs / 1e9:.2f}")
