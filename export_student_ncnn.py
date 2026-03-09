"""
Export student .pt models (trained with custom fixed_distillation_trainer) to NCNN.
Run this on any machine that has ultralytics installed.
"""

import pickle
import types
import torch
import torch.nn as nn


class _IdentityModule(nn.Module):
    """Replacement for unknown classes from fixed_distillation_trainer.
    Used as a no-op layer OR as a no-op forward hook on YOLO layers."""
    def __init__(self, *a, **k):
        super().__init__()  # sets _backward_hooks, _forward_hooks, etc.

    def __setstate__(self, state):
        # Initialize nn.Module internals first so _backward_hooks etc. exist,
        # then ignore the pickled state (we don't need the original attributes).
        super().__init__()

    def forward(self, x):
        return x

    def __call__(self, *args, **kwargs):
        # When used as a forward hook the signature is (module, input, output).
        # Return None to leave the layer's original output unchanged.
        if len(args) == 3 and isinstance(args[0], nn.Module):
            return None
        return super().__call__(*args, **kwargs)


class _PatchedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'fixed_distillation_trainer':
            return _IdentityModule
        return super().find_class(module, name)


# Build a patched pickle module accepted by torch.load's pickle_module parameter
patched_pickle = types.ModuleType('patched_pickle')
patched_pickle.__dict__.update(pickle.__dict__)
patched_pickle.Unpickler = _PatchedUnpickler


from ultralytics import YOLO

STUDENTS = [
    ('yolov8',  'colab_results/detect/yolov8/weights/best.pt'),
    ('yolo11',  'colab_results/detect/yolov11/weights/best.pt'),
    ('yolo26',  'colab_results/detect/yolov26/weights/best.pt'),
]

for name, path in STUDENTS:
    print(f"\n--- Exporting {name} student ---")
    try:
        # Load with patched pickle: unknown classes -> _IdentityModule (real nn.Module)
        # This avoids _Anything objects polluting the model's internal state.
        ckpt = torch.load(path, map_location='cpu', pickle_module=patched_pickle)

        # Save a clean checkpoint with no custom class references
        clean_path = path.replace('best.pt', 'best_clean.pt')
        torch.save({
            'epoch':        ckpt.get('epoch', -1),
            'best_fitness': ckpt.get('best_fitness', None),
            'model':        ckpt['model'],
            'ema':          None,
            'updates':      None,
            'optimizer':    None,
            'train_args':   {},
            'date':         None,
            'version':      None,
        }, clean_path)
        print(f"  Saved clean checkpoint: {clean_path}")

        # Load clean checkpoint with YOLO and export to NCNN
        model = YOLO(clean_path)
        model.export(format='ncnn', imgsz=640)
        print(f"  OK: NCNN model created")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {e}")
