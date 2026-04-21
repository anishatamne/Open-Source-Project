"""
SDXL 1.0 LoRA Fine-Tuning Pipeline — General Objects & Scenes
================================================================
Requirements:
    pip install torch torchvision diffusers transformers accelerate
    pip install peft bitsandbytes xformers datasets pillow tqdm
    pip install tensorboard safetensors

Directory structure:
    data/
        train/
            class_A/  img1.jpg  img2.jpg ...
            class_B/  img1.jpg  ...
        val/
            class_A/  ...
            class_B/  ...
    checkpoints/
    samples/
"""

import os
import math
import logging
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

from peft import LoraConfig, get_peft_model, TaskType
from torchvision import transforms
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)



class Config:
    # ── Model ──────────────────────────────────────────
    PRETRAINED_MODEL    = "stabilityai/stable-diffusion-xl-base-1.0"
    VAE_MODEL           = "madebyollin/sdxl-vae-fp16-fix"   # More stable VAE
    REVISION            = None

    # ── Dataset ────────────────────────────────────────
    DATA_DIR            = "./data"
    IMAGE_SIZE          = 1024          # SDXL native resolution
    CENTER_CROP         = True
    RANDOM_FLIP         = True

    # ── LoRA ───────────────────────────────────────────
    LORA_RANK           = 16            # 4–64; higher = more expressive
    LORA_ALPHA          = 32            # Usually 2× rank
    LORA_DROPOUT        = 0.05
    # Which UNet layers to inject LoRA into
    LORA_TARGET_MODULES = [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ]

    # ── Training ───────────────────────────────────────
    BATCH_SIZE          = 2             # Increase if VRAM > 16 GB
    GRAD_ACCUM_STEPS    = 4             # Effective batch = BATCH_SIZE × GRAD_ACCUM
    NUM_EPOCHS          = 50
    MAX_TRAIN_STEPS     = 5000          # Overrides epochs if hit first
    LR                  = 1e-4
    LR_SCHEDULER        = "cosine"      # constant | cosine | linear
    LR_WARMUP_STEPS     = 200
    ADAM_BETA1          = 0.9
    ADAM_BETA2          = 0.999
    ADAM_WEIGHT_DECAY   = 1e-2
    ADAM_EPSILON        = 1e-8
    MAX_GRAD_NORM       = 1.0
    MIXED_PRECISION     = "fp16"        # "no" | "fp16" | "bf16"
    GRADIENT_CHECKPOINTING = True

    # ── Noise / Diffusion ──────────────────────────────
    NOISE_OFFSET        = 0.1          # Improves dark/bright image generation
    SNR_GAMMA           = 5.0          # Min-SNR loss weighting (set None to disable)

    # ── Text prompts ───────────────────────────────────
    # These are auto-built from class folder names, but you can override per-class:
    PROMPT_TEMPLATE     = "a high quality photo of a {class_name}, detailed, sharp focus"
    NEGATIVE_PROMPT     = "blurry, low quality, distorted, watermark, text"

    # ── Validation ─────────────────────────────────────
    VAL_PROMPTS         = [
        "a high quality photo of a {class_name}, studio lighting",
        "a high quality photo of a {class_name}, outdoor scene",
    ]
    NUM_VALIDATION_IMAGES = 4
    VALIDATION_EVERY    = 500           # steps

    # ── Saving ─────────────────────────────────────────
    OUTPUT_DIR          = "./checkpoints"
    SAMPLE_DIR          = "./samples"
    SAVE_EVERY          = 500           # steps
    LOGGING_DIR         = "./logs"


cfg = Config()


# ══════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════
class SceneDataset(Dataset):
    """
    Folder-based dataset. Each subfolder = one class.
    Returns (pixel_values, prompt_string).
    """
    EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, split="train", image_size=1024,
                 center_crop=True, random_flip=True,
                 prompt_template=None):
        self.samples = []   # (image_path, class_name)
        split_dir = Path(root) / split

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.replace("_", " ")
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self.EXT:
                    self.samples.append((img_path, cls_name))

        self.prompt_template = prompt_template or cfg.PROMPT_TEMPLATE
        aug = [transforms.RandomHorizontalFlip()] if random_flip and split == "train" else []
        crop = transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size)

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
            *aug,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # ✅ FIXED HERE
        ])

        log.info(f"[{split}] {len(self.samples)} images | "
                 f"{len({s[1] for s in self.samples})} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        prompt = self.prompt_template.format(class_name=cls_name)
        return {
            "pixel_values": self.transform(img),
            "prompt": prompt,
            "class_name": cls_name,
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "prompts":      [b["prompt"] for b in batch],
        "class_names":  [b["class_name"] for b in batch],
    }


# ══════════════════════════════════════════════════════
# TEXT ENCODING  (SDXL uses two encoders)
# ══════════════════════════════════════════════════════
def encode_prompts(prompts, tokenizer_1, tokenizer_2,
                   text_encoder_1, text_encoder_2, device):
    """
    Returns (prompt_embeds, pooled_prompt_embeds) for SDXL.
    SDXL concatenates hidden states from CLIP-L and CLIP-G.
    """
    def _tokenize(tokenizer, prompts):
        return tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

    ids_1 = _tokenize(tokenizer_1, prompts)
    ids_2 = _tokenize(tokenizer_2, prompts)

    with torch.no_grad():
        enc_1_out = text_encoder_1(ids_1, output_hidden_states=True)
        enc_2_out = text_encoder_2(ids_2, output_hidden_states=True)

    # SDXL uses penultimate hidden states from both encoders
    hidden_1 = enc_1_out.hidden_states[-2]                 # (B, 77, 768)
    hidden_2 = enc_2_out.hidden_states[-2]                 # (B, 77, 1280)
    prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1) # (B, 77, 2048)
    pooled = enc_2_out[0]                                   # (B, 1280)

    return prompt_embeds, pooled


# ══════════════════════════════════════════════════════
# MIN-SNR LOSS WEIGHTING
# ══════════════════════════════════════════════════════
def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas / sqrt_one_minus) ** 2
    return snr


def snr_loss_weight(noise_scheduler, timesteps, gamma=5.0):
    snr = compute_snr(noise_scheduler, timesteps)
    weights = torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
    return weights


# ══════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════
@torch.no_grad()
def run_validation(unet, vae, text_enc_1, text_enc_2,
                   tok_1, tok_2, accelerator, step, class_names):
    log.info("Running validation...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        cfg.PRETRAINED_MODEL,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_enc_1,
        text_encoder_2=text_enc_2,
        tokenizer=tok_1,
        tokenizer_2=tok_2,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)
    for cls in class_names[:3]:   # Validate on first 3 classes
        for tmpl in cfg.VAL_PROMPTS:
            prompt = tmpl.format(class_name=cls)
            images = pipeline(
                prompt=prompt,
                negative_prompt=cfg.NEGATIVE_PROMPT,
                num_images_per_prompt=cfg.NUM_VALIDATION_IMAGES,
                num_inference_steps=25,
                guidance_scale=7.5,
            ).images
            for i, img in enumerate(images):
                fname = f"{cfg.SAMPLE_DIR}/step{step}_{cls.replace(' ','_')}_{i}.png"
                img.save(fname)

    del pipeline
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════
def main():
    # ── Accelerator ────────────────────────────────────
    proj_cfg = ProjectConfiguration(project_dir=cfg.OUTPUT_DIR,
                                    logging_dir=cfg.LOGGING_DIR)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        mixed_precision=cfg.MIXED_PRECISION,
        log_with="tensorboard",
        project_config=proj_cfg,
    )
    accelerator.init_trackers("sdxl-lora")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.SAMPLE_DIR,  exist_ok=True)

    # ── Load Models ────────────────────────────────────
    log.info("Loading SDXL components...")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.PRETRAINED_MODEL,
                                                    subfolder="scheduler")
    tokenizer_1 = CLIPTokenizer.from_pretrained(cfg.PRETRAINED_MODEL,
                                                subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.PRETRAINED_MODEL,
                                                subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        cfg.PRETRAINED_MODEL, subfolder="text_encoder",
        torch_dtype=torch.float16).requires_grad_(False)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        cfg.PRETRAINED_MODEL, subfolder="text_encoder_2",
        torch_dtype=torch.float16).requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(
        cfg.VAE_MODEL, torch_dtype=torch.float16).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(
        cfg.PRETRAINED_MODEL, subfolder="unet",
        torch_dtype=torch.float32)

    # Move frozen models to device
    text_encoder_1.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    vae.to(accelerator.device)

    if cfg.GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()

    # ── Inject LoRA into UNet ───────────────────────────
    log.info(f"Injecting LoRA | rank={cfg.LORA_RANK} | alpha={cfg.LORA_ALPHA}")
    lora_config = LoraConfig(
        r=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Dataset & DataLoader ───────────────────────────
    train_ds = SceneDataset(cfg.DATA_DIR, split="train",
                            image_size=cfg.IMAGE_SIZE,
                            center_crop=cfg.CENTER_CROP,
                            random_flip=cfg.RANDOM_FLIP)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=4, pin_memory=True, drop_last=True)

    all_classes = list({s[1] for s in train_ds.samples})

    # ── Optimizer & Scheduler ─────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=cfg.LR,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
        weight_decay=cfg.ADAM_WEIGHT_DECAY,
        eps=cfg.ADAM_EPSILON,
    )

    num_update_steps = math.ceil(len(train_dl) / cfg.GRAD_ACCUM_STEPS)
    total_steps = min(cfg.MAX_TRAIN_STEPS, cfg.NUM_EPOCHS * num_update_steps)

    lr_scheduler = get_scheduler(
        cfg.LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=cfg.LR_WARMUP_STEPS * cfg.GRAD_ACCUM_STEPS,
        num_training_steps=total_steps * cfg.GRAD_ACCUM_STEPS,
    )

    # ── Prepare with Accelerator ───────────────────────
    unet, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dl, lr_scheduler
    )

    # ── Training ───────────────────────────────────────
    global_step = 0
    log.info(f"Training | steps={total_steps} | device={accelerator.device}")

    for epoch in range(cfg.NUM_EPOCHS):
        unet.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}",
                    disable=not accelerator.is_local_main_process)

        for batch in pbar:
            with accelerator.accumulate(unet):

                # 1. Encode images to latents
                pixel_values = batch["pixel_values"].to(accelerator.device,
                                                        dtype=torch.float16)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 2. Sample noise
                noise = torch.randn_like(latents)
                if cfg.NOISE_OFFSET:
                    noise += cfg.NOISE_OFFSET * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1,
                        device=latents.device)

                B = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (B,), device=latents.device).long()

                # 3. Add noise (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 4. Encode text
                prompt_embeds, pooled_embeds = encode_prompts(
                    batch["prompts"],
                    tokenizer_1, tokenizer_2,
                    text_encoder_1, text_encoder_2,
                    accelerator.device,
                )
                prompt_embeds = prompt_embeds.to(dtype=unet.dtype)
                pooled_embeds = pooled_embeds.to(dtype=unet.dtype)

                # 5. SDXL add_time_ids (original resolution conditioning)
                add_time_ids = torch.tensor(
                    [[cfg.IMAGE_SIZE, cfg.IMAGE_SIZE,  # original H, W
                      0, 0,                            # crop top, left
                      cfg.IMAGE_SIZE, cfg.IMAGE_SIZE]  # target H, W
                    ] * B,
                    dtype=unet.dtype, device=accelerator.device
                )

                # 6. UNet forward pass
                model_pred = unet(
                    noisy_latents.to(dtype=unet.dtype),
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                # 7. Compute loss (v-prediction or epsilon)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: "
                                     f"{noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, loss.ndim)))  # per-sample

                # 8. Min-SNR weighting
                if cfg.SNR_GAMMA is not None:
                    weights = snr_loss_weight(noise_scheduler, timesteps, cfg.SNR_GAMMA)
                    weights = weights.to(loss.device)
                    loss = (loss * weights).mean()
                else:
                    loss = loss.mean()

                # 9. Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, unet.parameters()),
                        cfg.MAX_GRAD_NORM
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── Logging ───────────────────────────────
            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.detach().item()
                pbar.set_postfix(loss=f"{loss.detach().item():.4f}",
                                 lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                                 step=global_step)
                accelerator.log({"train/loss": loss.detach().item(),
                                 "train/lr": lr_scheduler.get_last_lr()[0]},
                                step=global_step)

                # ── Validation samples ─────────────────
                if (global_step % cfg.VALIDATION_EVERY == 0 and
                        accelerator.is_local_main_process):
                    run_validation(unet, vae, text_encoder_1, text_encoder_2,
                                   tokenizer_1, tokenizer_2,
                                   accelerator, global_step, all_classes)
                    unet.train()

                # ── Save checkpoint ────────────────────
                if global_step % cfg.SAVE_EVERY == 0:
                    save_lora_weights(unet, accelerator, global_step)

                if global_step >= total_steps:
                    break

        avg_loss = epoch_loss / max(len(train_dl), 1)
        log.info(f"Epoch {epoch+1} complete | avg_loss={avg_loss:.4f}")

        if global_step >= total_steps:
            log.info("Reached max training steps. Stopping.")
            break

    # ── Final save ────────────────────────────────────
    save_lora_weights(unet, accelerator, global_step, final=True)
    accelerator.end_training()
    log.info("Training complete!")


# ══════════════════════════════════════════════════════
# SAVE LoRA WEIGHTS
# ══════════════════════════════════════════════════════
def save_lora_weights(unet, accelerator, step, final=False):
    if not accelerator.is_local_main_process:
        return
    tag = "final" if final else f"step{step}"
    save_dir = Path(cfg.OUTPUT_DIR) / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(unet)
    # Extract only LoRA delta weights
    lora_state = {k: v for k, v in unwrapped.state_dict().items()
                  if "lora" in k.lower()}
    save_file(lora_state, save_dir / "lora_weights.safetensors")
    log.info(f"LoRA weights saved → {save_dir}/lora_weights.safetensors")


# ══════════════════════════════════════════════════════
# INFERENCE  — load LoRA and generate images
# ══════════════════════════════════════════════════════
def generate(
    lora_path: str,
    prompts: list[str],
    num_images: int = 4,
    steps: int = 30,
    guidance_scale: float = 7.5,
    lora_scale: float = 0.9,
    seed: int = 42,
    output_dir: str = "./generated",
):
    """
    Load fine-tuned LoRA weights and generate images.

    Args:
        lora_path       : Path to folder containing lora_weights.safetensors
        prompts         : List of text prompts
        num_images      : Images per prompt
        steps           : Denoising steps (20–50)
        guidance_scale  : CFG scale (5.0–10.0)
        lora_scale      : LoRA influence (0.0–1.0)
        seed            : Reproducibility seed
        output_dir      : Folder to save generated images
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading SDXL pipeline for inference...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.PRETRAINED_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    # Faster scheduler for inference
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    # Load LoRA weights
    log.info(f"Loading LoRA from {lora_path}")
    pipe.load_lora_weights(lora_path, weight_name="lora_weights.safetensors")
    pipe.fuse_lora(lora_scale=lora_scale)   # Fuse for faster inference

    generator = torch.Generator(device=device).manual_seed(seed)

    all_images = []
    for i, prompt in enumerate(prompts):
        log.info(f"[{i+1}/{len(prompts)}] Generating: {prompt}")
        images = pipe(
            prompt=prompt,
            negative_prompt=cfg.NEGATIVE_PROMPT,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=cfg.IMAGE_SIZE,
            width=cfg.IMAGE_SIZE,
        ).images

        for j, img in enumerate(images):
            fname = f"{output_dir}/prompt{i:02d}_img{j:02d}.png"
            img.save(fname)
            all_images.append(img)
            log.info(f"  Saved → {fname}")

    log.info(f"Done! {len(all_images)} images saved to {output_dir}")
    return all_images


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SDXL 1.0 LoRA Fine-Tuning")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--lora_path", type=str, default="./checkpoints/final",
                        help="Path to saved LoRA weights (for generate mode)")
    parser.add_argument("--prompts", nargs="+",
                        default=["a high quality photo of a cat on a table"],
                        help="Prompts for generation mode")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "train":
        main()
    else:
        generate(
            lora_path=args.lora_path,
            prompts=args.prompts,
            num_images=args.num_images,
            seed=args.seed,
        )
