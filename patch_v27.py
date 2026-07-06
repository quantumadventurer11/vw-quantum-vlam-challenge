import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/phase1_baseline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = []

for cell in nb['cells']:
    cid = cell.get('id', '?')
    src = ''.join(cell['source'])

    # ── Cell 15 (warmup): cast pixel_values to fp16 after .to("cuda:0") ──────
    if cid == 'warmup':
        old = (
            'warmup_inputs = processor(\n'
            '    warmup_sample["language"], warmup_sample["image"]\n'
            ').to("cuda:0")\n'
            '\n'
            'with torch.no_grad():'
        )
        new = (
            'warmup_inputs = processor(\n'
            '    warmup_sample["language"], warmup_sample["image"]\n'
            ').to("cuda:0")\n'
            '# FP16 model (P100 sm_60): processor returns float32 pixel_values;\n'
            '# patch_embed Conv2d bias is fp16 -> must cast before forward pass.\n'
            'if "pixel_values" in warmup_inputs:\n'
            '    warmup_inputs["pixel_values"] = warmup_inputs["pixel_values"].to(dtype=torch.float16)\n'
            '\n'
            'with torch.no_grad():'
        )
        assert old in src, f'warmup anchor not found. Lines with processor:\n' + '\n'.join(
            l for l in src.splitlines() if 'processor' in l or 'warmup_inputs' in l
        )
        src = src.replace(old, new, 1)
        cell['source'] = [src]
        patched.append(cid)

    # ── Cell 16 (inference-fn): cast pixel_values to fp16 after .to("cuda:0") ─
    elif cid == 'inference-fn':
        old = (
            '    inputs = processor(language, image).to("cuda:0")\n'
            '\n'
            '    pwr_before_w'
        )
        new = (
            '    inputs = processor(language, image).to("cuda:0")\n'
            '    if "pixel_values" in inputs:\n'
            '        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)\n'
            '\n'
            '    pwr_before_w'
        )
        assert old in src, f'inference-fn anchor not found. Lines with inputs:\n' + '\n'.join(
            l for l in src.splitlines() if 'inputs' in l or 'processor' in l
        )
        src = src.replace(old, new, 1)
        cell['source'] = [src]
        patched.append(cid)

assert set(patched) == {'warmup', 'inference-fn'}, f'Patched: {patched}'
print('Patched cells:', patched)

with open('notebooks/phase1_baseline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook saved.')
