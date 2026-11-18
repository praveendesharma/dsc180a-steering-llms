"""Compare the desperate and calm LoRA adapters with each other and with the emotion vectors.

Run from notebooks/ after lora.ipynb. Needs the extracted emotion vectors (see steering.ipynb).
"""
import json, numpy as np
from safetensors.numpy import load_file
run = "../emotion_vector_pipeline/outputs/emotion_extraction/ab1333f49024705c/emotion_vectors.npz"
with np.load(run) as s:
    names = s["emotion_names"].tolist(); V = s["unit_vectors"][:, s["layer_indices"].tolist().index(23)]
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
push = {}
for e in ["desperate", "calm"]:
    w = load_file(f"../notebooks/outputs/lora_adapters/{e}/adapter_model.safetensors")
    B = [v for k, v in w.items() if "lora_B" in k][0][:, 0].astype(np.float64)
    r = json.load(open(f"../results/lora/{e}_adapter.json"))
    push[e] = r["firing_by_emotion"][e] * B  # average push direction on its own emotion
    print(e, "firing range across emotions:", round(min(r["firing_by_emotion"].values()), 2), "to", round(max(r["firing_by_emotion"].values()), 2))
d, c = names.index("desperate"), names.index("calm")
print("cos(push_desperate, push_calm):", round(cos(push["desperate"], push["calm"]), 3))
diff = push["desperate"] - push["calm"]
print("cos(push_desp - push_calm, v_desp - v_calm):", round(cos(diff, V[d] - V[c]), 3))
ranks = sorted(((cos(diff, V[i]), n) for i, n in enumerate(names)), reverse=True)
print("diff vs each emotion vector:", [(n, round(x, 3)) for x, n in ranks])
