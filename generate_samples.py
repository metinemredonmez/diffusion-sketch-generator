#!/usr/bin/env python3
"""
Generate samples from trained models and compute FID/KID metrics.
"""

import os
import sys
import json
import ndjson
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
import math
from einops import rearrange, repeat

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# ============================================================================
# Model Definition (same as training)
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)

        x = residual + out
        x = x + self.ff(self.norm2(x))

        return x

class SketchDiffusionModel(nn.Module):
    def __init__(self, seq_len=200, input_dim=3, hidden_dim=256, num_layers=4, num_heads=4):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.category_embedding = nn.Embedding(3, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, heads=num_heads, dim_head=hidden_dim//num_heads)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, category):
        x = self.input_proj(x)
        x = x + self.pos_embedding

        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1)

        c_emb = self.category_embedding(category)
        c_emb = c_emb.unsqueeze(1)

        x = x + t_emb + c_emb

        for block in self.transformer_blocks:
            x = block(x)

        x = self.output_proj(x)

        return x

# ============================================================================
# Diffusion Sampling
# ============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiffusionSampler:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.to(device)

    @torch.no_grad()
    def p_sample(self, x, t, category):
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)

        predicted_noise = self.model(x, t, category)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape, category):
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, category)

        return x

# ============================================================================
# Visualization Functions
# ============================================================================

def sequence_to_strokes(sequence):
    strokes = []
    current_stroke_x = []
    current_stroke_y = []

    x, y = 0, 0

    for dx, dy, pen in sequence:
        if dx == 0 and dy == 0 and pen == 0:
            continue

        x += dx
        y += dy

        if pen > 0.5:
            current_stroke_x.append(x)
            current_stroke_y.append(y)
        else:
            if len(current_stroke_x) > 0:
                strokes.append([current_stroke_x.copy(), current_stroke_y.copy()])
                current_stroke_x = []
                current_stroke_y = []

    if len(current_stroke_x) > 0:
        strokes.append([current_stroke_x, current_stroke_y])

    return strokes

def draw_sketch(strokes, size=(256, 256), line_width=2):
    img = Image.new('L', size, 255)
    draw = ImageDraw.Draw(img)

    all_x = []
    all_y = []
    for stroke in strokes:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])

    if len(all_x) == 0:
        return img

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    width = max_x - min_x
    height = max_y - min_y
    if width == 0:
        width = 1
    if height == 0:
        height = 1

    padding = 20
    scale = min((size[0] - 2*padding) / width,
                (size[1] - 2*padding) / height)

    for stroke in strokes:
        if len(stroke[0]) < 2:
            continue

        points = []
        for x, y in zip(stroke[0], stroke[1]):
            x_scaled = (x - min_x) * scale + padding
            y_scaled = (y - min_y) * scale + padding
            points.append((x_scaled, y_scaled))

        if len(points) >= 2:
            draw.line(points, fill=0, width=line_width)

    return img

def create_generation_gif(sequence, filename, size=(256, 256), duration=50):
    frames = []

    step_size = max(1, len(sequence) // 40)
    for i in range(step_size, len(sequence) + 1, step_size):
        partial_seq = sequence[:i]
        strokes = sequence_to_strokes(partial_seq)
        if len(strokes) > 0:
            img = draw_sketch(strokes, size=size)
            frames.append(img)

    if len(frames) == 0:
        img = Image.new('L', size, 255)
        frames.append(img)

    for _ in range(5):
        frames.append(frames[-1])

    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f'Saved GIF: {filename}')

@torch.no_grad()
def generate_samples(model, sampler, category_id, num_samples=10):
    model.eval()
    category = torch.full((num_samples,), category_id, device=device, dtype=torch.long)
    samples = sampler.sample(shape=(num_samples, 200, 3), category=category)
    return samples.cpu().numpy()

# ============================================================================
# Data Loading for FID
# ============================================================================

def load_ndjson_sketches(filepath, indices=None):
    sketches = []
    with open(filepath, 'r') as f:
        data = ndjson.load(f)

    if indices is not None:
        data = [data[i] for i in indices if i < len(data)]

    for item in data:
        if 'drawing' in item:
            sketches.append(item['drawing'])

    return sketches

def strokes_to_sequence(strokes, max_len=200):
    sequence = []
    prev_x, prev_y = 0, 0

    for stroke_idx, stroke in enumerate(strokes):
        xs, ys = stroke[0], stroke[1]

        if len(xs) == 0:
            continue

        if stroke_idx == 0:
            prev_x, prev_y = xs[0], ys[0]
            sequence.append([xs[0], ys[0], 1])
        else:
            dx = xs[0] - prev_x
            dy = ys[0] - prev_y
            sequence.append([dx, dy, 0])

        for i in range(1, len(xs)):
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            sequence.append([dx, dy, 1])

        prev_x = xs[-1]
        prev_y = ys[-1]

    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        while len(sequence) < max_len:
            sequence.append([0, 0, 0])

    return np.array(sequence, dtype=np.float32)

def normalize_sequence(seq):
    xy = seq[:, :2].copy()

    non_pad = ~((xy[:, 0] == 0) & (xy[:, 1] == 0) & (seq[:, 2] == 0))

    if np.sum(non_pad) > 0:
        xy_valid = xy[non_pad]
        mean = xy_valid.mean(axis=0)
        std = xy_valid.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)

        xy[non_pad] = (xy_valid - mean) / std

    result = np.concatenate([xy, seq[:, 2:3]], axis=1)
    return result

def prepare_images_for_fid(sequences, output_dir, size=(299, 299)):
    os.makedirs(output_dir, exist_ok=True)

    for idx, seq in enumerate(tqdm(sequences, desc='Rendering images')):
        strokes = sequence_to_strokes(seq)
        img = draw_sketch(strokes, size=size)
        img_rgb = Image.new('RGB', img.size, (255, 255, 255))
        img_rgb.paste(img)
        img_rgb.save(f'{output_dir}/{idx:05d}.png')

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./fid_eval', exist_ok=True)

    categories = ['cat', 'bus', 'rabbit']
    results = {}

    for idx, category in enumerate(categories):
        print(f'\n{"="*60}')
        print(f'Processing category: {category.upper()}')
        print(f'{"="*60}')

        # Load model
        model_path = f'./models/{category}_final.pt'
        if not os.path.exists(model_path):
            print(f'Model not found: {model_path}')
            continue

        model = SketchDiffusionModel(
            seq_len=200,
            input_dim=3,
            hidden_dim=256,
            num_layers=4,
            num_heads=4
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        sampler = DiffusionSampler(model, timesteps=1000)

        # Generate samples for visualization
        print(f'\nGenerating samples for {category}...')
        samples = generate_samples(model, sampler, idx, num_samples=20)

        # Create visualization grid
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle(f'Generated {category.capitalize()} Sketches', fontsize=16)

        for i, ax in enumerate(axes.flat):
            strokes = sequence_to_strokes(samples[i])
            img = draw_sketch(strokes)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'./results/{category}_generated_samples.png', dpi=150)
        plt.close()

        # Create GIFs
        for i in range(3):
            create_generation_gif(
                samples[i],
                f'./results/{category}_generation_{i+1}.gif',
                size=(256, 256),
                duration=50
            )

        print(f'Saved visualizations for {category}')

        # Load test data for FID
        data_path = f'./data/{category}.ndjson'
        subset_path = f'./subset/{category}/indices.json'

        if os.path.exists(subset_path):
            with open(subset_path, 'r') as f:
                indices_data = json.load(f)
            test_indices = indices_data.get('test', [])[:500]
        else:
            test_indices = list(range(500))

        test_sketches = load_ndjson_sketches(data_path, test_indices)

        # Compute FID/KID
        print(f'\nComputing FID/KID for {category}...')

        # Prepare real images
        real_dir = f'./fid_eval/{category}/real'
        test_sequences = []
        for sketch in tqdm(test_sketches[:500], desc='Processing test sketches'):
            seq = strokes_to_sequence(sketch, max_len=200)
            seq = normalize_sequence(seq)
            test_sequences.append(seq)

        prepare_images_for_fid(test_sequences, real_dir)

        # Generate and prepare fake images
        fake_dir = f'./fid_eval/{category}/fake'
        print('Generating samples for FID...')
        generated_samples = []

        batch_size = 25
        num_batches = 500 // batch_size

        for _ in tqdm(range(num_batches), desc='Generating batches'):
            batch_samples = generate_samples(model, sampler, idx, num_samples=batch_size)
            generated_samples.extend(batch_samples)

        prepare_images_for_fid(generated_samples, fake_dir)

        # Compute metrics
        try:
            from cleanfid import fid

            print('Computing FID...')
            fid_score = fid.compute_fid(real_dir, fake_dir, mode='clean', num_workers=0)

            print('Computing KID...')
            kid_score = fid.compute_kid(real_dir, fake_dir, mode='clean', num_workers=0)

            results[category] = {
                'FID': float(fid_score),
                'KID': float(kid_score)
            }

            print(f'\n{category.upper()} Results:')
            print(f'  FID: {fid_score:.4f}')
            print(f'  KID: {kid_score:.4f}')
        except Exception as e:
            print(f'Error computing FID/KID: {e}')
            results[category] = {
                'FID': None,
                'KID': None,
                'error': str(e)
            }

    # Print final results
    print('\n' + '='*60)
    print('FINAL EVALUATION RESULTS')
    print('='*60)
    print(f'{"Category":<15} {"FID":<15} {"KID":<15}')
    print('-'*60)

    for category in ['cat', 'bus', 'rabbit']:
        if category in results:
            fid_val = results[category].get('FID', 'N/A')
            kid_val = results[category].get('KID', 'N/A')
            if fid_val is not None:
                print(f'{category.capitalize():<15} {fid_val:<15.4f} {kid_val:<15.4f}')
            else:
                print(f'{category.capitalize():<15} {"Error":<15} {"Error":<15}')

    print('='*60)

    # Save results
    with open('./results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nAll done! Results saved to ./results/')
