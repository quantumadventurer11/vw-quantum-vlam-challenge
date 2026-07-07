"""patch_v28.py — insert fix-predict-action cell after '0c0409ba'

Root cause (v27):
  predict_action appends token 29871 to input_ids but does NOT extend the
  attention_mask that arrives via **kwargs.  In the multimodal forward pass,
  multimodal_embeddings gains one extra token (N+1+n_patches) while
  multimodal_attention_mask stays one short (N+n_patches), producing:

    RuntimeError: The size of tensor a (259) must match the size of tensor b
    (258) at non-singleton dimension 3   [attn_weights + causal_mask]

Fix: monkey-patch model.predict_action to mirror the same condition and extend
attention_mask by 1 whenever input_ids will be extended by 1.
"""
import json, sys

sys.stdout.reconfigure(encoding="utf-8")

FIX_CELL_ID = "fix-predict-action"
INSERT_AFTER = "0c0409ba"   # model-diagnostics cell

FIX_SOURCE = [
    "# ── Cell 10b: Fix predict_action attention_mask sync (v28) ────────────────────\n",
    "# predict_action conditionally appends token 29871 to input_ids before calling\n",
    "# generate(), but never extends the attention_mask passed via **kwargs.\n",
    "# Inside PrismaticForConditionalGeneration.forward() the multimodal expansion:\n",
    "#   multimodal_embeddings      = 1 + n_patches + (N+1−1) = N+1+n_patches tokens\n",
    "#   multimodal_attention_mask  = 1 + n_patches + (N−1)   = N+n_patches tokens\n",
    "# → off-by-one → RuntimeError: tensor a (259) != tensor b (258) dim 3.\n",
    "# Fix: mirror the same condition and extend attention_mask by 1 when needed.\n",
    "import types as _types\n",
    "\n",
    "def _predict_action_patched(self, input_ids=None, unnorm_key=None, **kwargs):\n",
    "    if not torch.all(input_ids[:, -1] == 29871):\n",
    "        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:\n",
    "            am = kwargs['attention_mask']\n",
    "            kwargs['attention_mask'] = torch.cat(\n",
    "                [am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)],\n",
    "                dim=1,\n",
    "            )\n",
    "    return type(self).predict_action(self, input_ids=input_ids, unnorm_key=unnorm_key, **kwargs)\n",
    "\n",
    "model.predict_action = _types.MethodType(_predict_action_patched, model)\n",
    "print('predict_action patched: attention_mask sync fix applied (v28).')\n",
]

with open("notebooks/phase1_baseline.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Guard: don't double-insert
if any(c.get("id") == FIX_CELL_ID for c in nb["cells"]):
    print(f"Cell '{FIX_CELL_ID}' already present — skipping insert.")
else:
    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("id") == INSERT_AFTER:
            insert_idx = i + 1
            break
    assert insert_idx is not None, f"Anchor cell '{INSERT_AFTER}' not found"

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": FIX_CELL_ID,
        "metadata": {},
        "outputs": [],
        "source": FIX_SOURCE,
    }
    nb["cells"].insert(insert_idx, new_cell)
    print(f"Inserted cell '{FIX_CELL_ID}' after '{INSERT_AFTER}' (index {insert_idx}).")

with open("notebooks/phase1_baseline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
