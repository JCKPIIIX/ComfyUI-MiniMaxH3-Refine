"""MiniMax H3 Refine (fl2va)

Prend une vidéo brouillon (frames + audio) générée par MiniMax H3 avec trop peu
de steps, et prépare tout ce qu'il faut pour la corriger avec un sampler à faible
denoise, sur le checkpoint fl2va :

  - un latent AV (NestedTensor : vidéo [1,24,T,H/16,W/16] + audio [1,32,2,T40])
    obtenu en encodant le brouillon avec le VAE vidéo et le VAE audio de H3 ;
  - un conditioning positif identique à celui de MiniMaxH3ImageToVideo
    (prompt Qwen3-VL + keyframes optionnelles propres, minimax_frame_count).

Installation : copier ce fichier dans ComfyUI/custom_nodes/ puis redémarrer.

Chaîne conseillée :
  GetVideoComponents -> [ce nœud] -> BasicGuider (cfg 1) -> SamplerCustomAdvanced
  (RandomNoise, KSamplerSelect, BasicScheduler avec denoise BAS) -> séparation AV
  -> VAEDecode (vidéo) + décodage VAE audio.

Denoise : avec un shift vidéo de 12, le denoise utile est très bas (voir le
README du pack : environ 0.03 à 0.12). Commencer par 0.01 (test
d'identité), puis balayer.
"""

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io

CANVAS_MULTIPLE = 32
FPS = 24
AUDIO_LATENT_FPS = 40
DEFAULT_AUDIO_SR = 32000


# --------------------------------------------------------------------------
# Géométrie temporelle de H3 (identique au nœud officiel nodes_minimax_h3.py)
# --------------------------------------------------------------------------

def snap_frame_count(n):
    """Plus grande longueur valide (17k+5) <= n."""
    if n < 5:
        raise ValueError(
            "MiniMax H3 Refine : il faut au moins 5 frames dans la vidéo brouillon (%d reçues)." % n)
    return ((n - 5) // 17) * 17 + 5


def video_latent_t(frame_count):
    if frame_count <= 1:
        return 1
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def audio_latent_t(frame_count):
    return round(frame_count / FPS * AUDIO_LATENT_FPS)


def _round32(v):
    return max(CANVAS_MULTIPLE, round(v / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _resample_fps(frames, source_fps):
    """Rééchantillonne (plus proche voisin) vers 24 fps si le brouillon a un autre fps."""
    if source_fps <= 0 or abs(source_fps - FPS) < 0.01:
        return frames
    n_src = frames.shape[0]
    n_dst = max(1, int(n_src / source_fps * FPS))
    idx = torch.clamp((torch.arange(n_dst) * (source_fps / FPS)).long(), max=n_src - 1)
    return frames[idx]


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

def _prepare_waveform(audio, vae_sr, audio_t):
    """audio dict -> waveform [1, 2, audio_t * (vae_sr // 40)] au sample rate du VAE."""
    wf = audio["waveform"][:1]  # [1, C, L]
    sr = audio["sample_rate"]
    if sr != vae_sr:
        wf = torchaudio.functional.resample(wf, sr, vae_sr)
    if wf.shape[1] == 1:
        wf = wf.repeat(1, 2, 1)
    elif wf.shape[1] > 2:
        wf = wf[:, :2]
    target = audio_t * (vae_sr // AUDIO_LATENT_FPS)
    length = wf.shape[-1]
    if length >= target:
        wf = wf[..., :target]
    else:
        wf = torch.nn.functional.pad(wf, (0, target - length))
    return wf


def _encode_audio(audio_vae, wf, audio_t):
    """Encode en latent [1, 32, 2, audio_t].

    La convention d'entrée de audio_vae.encode a varié pendant le développement
    du support H3 ([B, 2, L] vs [B, L, 2]) : on essaie les deux et on valide la
    forme de sortie.
    """
    last_error = None
    z = None
    for candidate in (wf, wf.movedim(1, -1)):
        try:
            out = audio_vae.encode(candidate)
        except Exception as e:  # layout rejeté par cette version du VAE
            last_error = e
            continue
        if out.ndim == 4 and out.shape[1] == 32 and out.shape[2] == 2:
            z = out
            break
    if z is None:
        raise RuntimeError(
            "MiniMax H3 Refine : impossible d'encoder l'audio avec ce VAE audio "
            "(attendu : latent [B, 32, 2, T]). Dernière erreur : %s" % last_error)

    t = z.shape[-1]
    if abs(t - audio_t) > 2:
        print("[MiniMaxH3Refine] attention : latent audio de %d pas, attendu %d" % (t, audio_t))
    if t > audio_t:
        z = z[..., :audio_t]
    elif t < audio_t:
        z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], audio_t - t)], dim=-1)
    return z[:1]


# --------------------------------------------------------------------------
# Nœud
# --------------------------------------------------------------------------

class MiniMaxH3RefineFL2VA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RefineFL2VA",
            display_name="MiniMax H3 Refine (fl2va)",
            category="model/conditioning/minimax",
            description=(
                "Encode une vidéo+audio brouillon en latent AV H3 et construit le conditioning "
                "fl2va pour la raffiner avec un sampler à faible denoise (≈0.03-0.12)."),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="VAE vidéo H3"),
                io.Vae.Input("audio_vae", tooltip="VAE audio H3"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="Même prompt (structuré) que celui du brouillon, soundscape inclus."),
                io.Image.Input("images", tooltip="Frames de la vidéo brouillon (24 fps idéalement)."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Audio du brouillon. Absent = silence (l'audio sera alors à peu près perdu à bas denoise)."),
                io.Float.Input("source_fps", default=24.0, min=1.0, max=120.0, step=0.01,
                               tooltip="FPS de la vidéo brouillon. Rééchantillonné à 24 fps si différent."),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="0 = taille du brouillon arrondie au multiple de 32."),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="0 = taille du brouillon arrondie au multiple de 32."),
                io.Image.Input("first_frame", optional=True,
                               tooltip="Keyframe PROPRE (pas du brouillon) ancrée à la frame 0, réinjectée à chaque step."),
                io.Image.Input("last_frame", optional=True,
                               tooltip="Keyframe PROPRE ancrée à la dernière frame."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.Int.Output(display_name="frames"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, images, audio=None, source_fps=24.0,
                width=0, height=0, first_frame=None, last_frame=None) -> io.NodeOutput:
        # ---- frames : fps -> 24, longueur 17k+5, taille multiple de 32 ----
        frames = _resample_fps(images[..., :3], source_fps)
        n = snap_frame_count(frames.shape[0])
        frames = frames[:n]

        w = width if width > 0 else _round32(frames.shape[2])
        h = height if height > 0 else _round32(frames.shape[1])
        if w % CANVAS_MULTIPLE or h % CANVAS_MULTIPLE:
            raise ValueError("MiniMax H3 Refine : width et height doivent être des multiples de 32.")
        frames = _resize(frames, w, h, "disabled")

        # ---- latent vidéo ----
        z_video = vae.encode(frames)[:1]
        expected = (24, video_latent_t(n), h // 16, w // 16)
        got = (z_video.shape[1], z_video.shape[2], z_video.shape[3], z_video.shape[4])
        if got != expected:
            raise RuntimeError(
                "MiniMax H3 Refine : latent vidéo inattendu %s, attendu %s "
                "(n=%d frames, %dx%d). Vérifie que c'est bien le VAE vidéo H3." % (got, expected, n, w, h))

        # ---- latent audio ----
        audio_t = audio_latent_t(n)
        vae_sr = getattr(audio_vae, "audio_sample_rate", DEFAULT_AUDIO_SR)
        if audio is None:
            wf = torch.zeros([1, 2, audio_t * (vae_sr // AUDIO_LATENT_FPS)])
        else:
            wf = _prepare_waveform(audio, vae_sr, audio_t)
        z_audio = _encode_audio(audio_vae, wf, audio_t)

        device = comfy.model_management.intermediate_device()
        latent = {"samples": comfy.nested_tensor.NestedTensor((
            z_video.to(device=device, dtype=torch.float32),
            z_audio.to(device=device, dtype=torch.float32),
        ))}

        # ---- conditioning (même construction que MiniMaxH3ImageToVideo) ----
        kf_images = []
        keyframes = []
        if first_frame is not None:
            img = _resize(first_frame[:1], w, h, "disabled")
            kf_images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            img = _resize(last_frame[:1], w, h, "center")
            kf_images.append(img)
            keyframes.append({"resolved_frame_index": n - 1, "image": img})

        tokens = clip.tokenize(prompt, images=kf_images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": n,
            })

        return io.NodeOutput(cond, latent, n)


class MiniMaxH3RefineExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3RefineFL2VA]


async def comfy_entrypoint() -> MiniMaxH3RefineExtension:
    return MiniMaxH3RefineExtension()
