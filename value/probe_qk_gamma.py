"""Follow-up: probe 1 showed plain's logit spread is ~4.9e-7, but the embedding
norm collapse (7.93 -> 0.40) only predicts a x204 drop, not x4e6. So something
ELSE also collapsed. Check the two candidates:

  (a) the q/k projection weights themselves (W_q, W_k inside hand_attn)
  (b) LayerNorm's learnable gain gamma (C/BCD only) — pre-LN pins the RMS to 1
      only if gamma stays ~1; a shrinking gamma re-opens the same door.
"""


import bootstrap  # noqa: F401

import torch

from value_model import ValueNet, D
from value_model_v2 import ValueNetV2

CKPT = bootstrap.CKPT_DIR
FACTORY = {
    "plain":   lambda: ValueNet(extra_dim=0),
    "featref": lambda: ValueNet(extra_dim=1),
    "B":       lambda: ValueNetV2(split_emb=True),
    "C":       lambda: ValueNetV2(pre_ln=True, bag_mean=True),
    "D":       lambda: ValueNetV2(residual=True, own_count=True),
    "BCD":     lambda: ValueNetV2(split_emb=True, pre_ln=True, bag_mean=True,
                                  residual=True, own_count=True),
}

print(f"  {'arm':10s} {'‖W_q‖':>9s} {'‖W_k‖':>9s} {'‖W_v‖':>9s} {'γ(LN)':>8s} "
      f"{'emb‖row‖':>9s} {'預測logit尺度':>14s}")

ref = {}
for name in ["plain", "featref", "B", "C", "D", "BCD"]:
    net = FACTORY[name]()
    sd = torch.load(f"{CKPT}/value_ab_{name}.pt", map_location="cpu")
    net.load_state_dict(sd["model"]); net.eval()
    W = net.hand_attn.in_proj_weight
    nq, nk, nv = (float(W[:D].norm()), float(W[D:2*D].norm()), float(W[2*D:].norm()))
    g = float(net.tok_ln.weight.abs().mean()) if getattr(net, "tok_ln", None) is not None else float("nan")
    emb = net.hand_emb if isinstance(net, ValueNetV2) else net.card_emb
    er = float(emb.weight[1:].norm(dim=1).mean())
    tok = er * (g if g == g else 1.0) if not isinstance(net, ValueNetV2) or net.tok_ln is None else 8.0 * g
    scale = (tok ** 2) * nq * nk / (D ** 0.5) / (D ** 0.5) / 4 ** 0.5
    ref[name] = scale
    print(f"  {name:10s} {nq:9.3f} {nk:9.3f} {nv:9.3f} "
          f"{g:8.3f} {er:9.3f} {scale:14.3e}")

for name in ["plain", "C"]:
    torch.manual_seed(0)
    net = FACTORY[name]()
    W = net.hand_attn.in_proj_weight
    g = float(net.tok_ln.weight.abs().mean()) if getattr(net, "tok_ln", None) is not None else float("nan")
    emb = net.hand_emb if isinstance(net, ValueNetV2) else net.card_emb
    print(f"  {'init-'+name:10s} {float(W[:D].norm()):9.3f} {float(W[D:2*D].norm()):9.3f} "
          f"{float(W[2*D:].norm()):9.3f} {g:8.3f} "
          f"{float(emb.weight[1:].norm(dim=1).mean()):9.3f}")

print(f"\n  分解 plain 的 logit 塌陷(vs C):")
p_net, c_net = FACTORY["plain"](), FACTORY["C"]()
p_net.load_state_dict(torch.load(f"{CKPT}/value_ab_plain.pt", map_location="cpu")["model"])
c_net.load_state_dict(torch.load(f"{CKPT}/value_ab_C.pt", map_location="cpu")["model"])
tok_p = float(p_net.card_emb.weight[1:].norm(dim=1).mean())
tok_c = 8.0 * float(c_net.tok_ln.weight.abs().mean())
qk_p = float(p_net.hand_attn.in_proj_weight[:D].norm()) * float(p_net.hand_attn.in_proj_weight[D:2*D].norm())
qk_c = float(c_net.hand_attn.in_proj_weight[:D].norm()) * float(c_net.hand_attn.in_proj_weight[D:2*D].norm())
print(f"    token‖x‖ 這一項(平方律):    ×{(tok_c/tok_p)**2:12.1f}")
print(f"    W_q·W_k 這一項:              ×{qk_c/qk_p:12.1f}")
print(f"    兩項相乘(預測總倍率):        ×{(tok_c/tok_p)**2 * qk_c/qk_p:12.1f}")
print(f"    probe 1 實測 logit 幅寬倍率:  ×     3967068.3")
