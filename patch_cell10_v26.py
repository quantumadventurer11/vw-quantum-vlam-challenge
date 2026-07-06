import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/phase1_baseline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

PATCH_BLOCK = r"""
# ── Numpy bridge patches (torch 2.2.2+cu118 + numpy 2.x) ─────────────────────
# torch 2.2.2 C extensions compiled against numpy 1.x ABI; numpy 2.x on Kaggle
# disables the bridge entirely (torch.from_numpy / tensor.numpy both raise
# RuntimeError: Numpy is not available).
# v25 failure: processor.__init__ was patched (above), but TVF.to_tensor called
# from processor(image) at warmup also calls torch.from_numpy internally.
# (1) TVF.to_tensor: PIL→tensor via frombuffer (no from_numpy needed)
import torchvision.transforms.functional as _TVF
_orig_tvf_to_tensor = _TVF.to_tensor
def _numpy_free_to_tensor(pic):
    if isinstance(pic, torch.Tensor):
        return pic
    if hasattr(pic, "mode"):  # PIL Image
        _I16_KEY = "I;16" if _sys.byteorder == "little" else "I;16B"
        _dtype_map = {"I": torch.int32, _I16_KEY: torch.int16, "F": torch.float32}
        _dtype = _dtype_map.get(pic.mode, torch.uint8)
        _raw = torch.frombuffer(bytearray(pic.tobytes()), dtype=_dtype)
        _n_ch = len(pic.getbands())
        _raw = _raw.reshape(pic.height, pic.width, _n_ch).permute(2, 0, 1).contiguous()
        if pic.mode == "1":
            return (_raw > 0).to(dtype=torch.float32)
        if _dtype == torch.uint8:
            return _raw.to(dtype=torch.float32).div_(255.0)
        return _raw.to(dtype=torch.float32)
    return _orig_tvf_to_tensor(pic)
_TVF.to_tensor = _numpy_free_to_tensor
# (2) tensor.numpy: fall back via tolist->np.array when bridge disabled
_orig_tensor_numpy = torch.Tensor.numpy
def _patched_tensor_numpy(self, *, force=False):
    try:
        return _orig_tensor_numpy(self, force=force)
    except RuntimeError as _e:
        if "Numpy is not available" in str(_e):
            return np.array(self.detach().cpu().tolist())
        raise
torch.Tensor.numpy = _patched_tensor_numpy
print("Numpy bridge patches applied (TVF.to_tensor + Tensor.numpy).")
"""

patched = False
for cell in nb['cells']:
    if cell.get('id') != 'load-model':
        continue
    src = ''.join(cell['source'])
    old_anchor = 'torch.cuda.reset_peak_memory_stats()\n\nif USE_INT8:'
    new_anchor = 'torch.cuda.reset_peak_memory_stats()\n' + PATCH_BLOCK + '\nif USE_INT8:'
    assert old_anchor in src, f'Anchor not found! Lines around reset_peak:\n' + '\n'.join(
        l for l in src.splitlines() if 'reset_peak' in l or 'USE_INT8' in l
    )
    new_src = src.replace(old_anchor, new_anchor, 1)
    cell['source'] = [new_src]
    print(f'Cell patched. New line count: {len(new_src.splitlines())}')
    patched = True
    break

assert patched, 'load-model cell not found'

with open('notebooks/phase1_baseline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook saved.')
