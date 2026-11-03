"""ComfyUI-MiniMaxH3-Refine

Nœud de raffinement vidéo+audio pour MiniMax H3 (checkpoint fl2va).
Nécessite une version de ComfyUI avec le support natif de MiniMax H3
(PR Comfy-Org/ComfyUI #15224) et l'API `comfy_api.latest`.
"""

from .minimax_h3_refine import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
