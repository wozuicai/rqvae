#!/usr/bin/env python3
"""
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH python3 /root/judge/rqvae_title_pipeline_generated_rubrics.py \
    --input_json /root/judge/generated_rubrics_6.30query_new_sg.jsonl \
    --output_dir /root/judge/rqvae_outputs_generated_rubrics \
    --init_codebook kmeans \
    --kmeans_init_max_samples 50000 \
    --kmeans_init_n_init 10 \
    --kmeans_init_max_iter 300 \
    --epochs 60 \
    --batch_size 512 \
    --codebook_sizes 64 128 256 \
    --num_codebooks 2 3 \
    --latent_dims 128 256 \
    --beta_commit 0.1 0.25 0.4 \
    --seeds 42 123 2026 \
    --normalize
"""
import argparse
import dataclasses
import json
import logging
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


LOGGER = logging.getLogger("rqvae_title_generated_rubrics")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_texts_with_sentence_transformer(model_dir: str, texts: List[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_dir)
    LOGGER.info("SentenceTransformer device: %s", model.device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


class EmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class MLP(nn.Module):
    def __init__(self, dims: List[int], last_activation: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if (not is_last) or last_activation:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        dim: int,
        beta_commit: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta_commit = beta_commit
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        embed = torch.randn(num_codebooks, codebook_size, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        commit_loss = torch.tensor(0.0, device=z.device)

        all_indices: List[torch.Tensor] = []
        residual_norms: List[torch.Tensor] = [torch.norm(residual, dim=-1).mean().detach()]

        for l in range(self.num_codebooks):
            codebook = self.embed[l]
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ codebook.t()
                + codebook.pow(2).sum(dim=1)
            )
            idx = torch.argmin(dist, dim=1)
            all_indices.append(idx)

            q = F.embedding(idx, codebook)
            quantized_sum = quantized_sum + q
            residual = residual - q
            residual_norms.append(torch.norm(residual, dim=-1).mean().detach())

            commit_loss = commit_loss + F.mse_loss(z - (quantized_sum - q).detach(), q.detach())

            if self.training:
                self._ema_update(l, residual + q, idx)

        z_q = z + (quantized_sum - z).detach()
        commit_loss = self.beta_commit * commit_loss / self.num_codebooks
        return z_q, commit_loss, all_indices, residual_norms

    @torch.no_grad()
    def _ema_update(self, level: int, x_target: torch.Tensor, idx: torch.Tensor) -> None:
        one_hot = F.one_hot(idx, num_classes=self.codebook_size).type_as(x_target)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ x_target

        self.cluster_size[level].mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.embed_avg[level].mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.cluster_size[level].sum()
        cluster_size = (
            (self.cluster_size[level] + self.ema_eps)
            / (n + self.codebook_size * self.ema_eps)
            * n
        )
        self.embed[level].copy_(self.embed_avg[level] / cluster_size.unsqueeze(1))


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        beta_commit: float,
        ema_decay: float,
    ) -> None:
        super().__init__()
        self.encoder = MLP([input_dim, 512, latent_dim])
        self.rvq = ResidualVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            dim=latent_dim,
            beta_commit=beta_commit,
            ema_decay=ema_decay,
        )
        self.decoder = MLP([latent_dim, 512, input_dim], last_activation=False)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        z = self.encoder(x)
        z_q, commit_loss, indices, residual_norms = self.rvq(z)
        x_hat = self.decoder(z_q)
        return {
            "x_hat": x_hat,
            "commit_loss": commit_loss,
            "indices": indices,
            "residual_norms": residual_norms,
        }


@dataclasses.dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    cosine_weight: float = 0.1
    use_cosine_loss: bool = True
    grad_clip: float = 1.0


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(x, y, dim=-1)).mean()


def run_epoch(
    model: RQVAE,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total = Counter()
    for batch in loader:
        x = batch.to(device)
        out = model(x)
        x_hat = out["x_hat"]
        recon_mse = F.mse_loss(x_hat, x)
        recon_cos = cosine_loss(x_hat, x)
        recon = recon_mse + (cfg.cosine_weight * recon_cos if cfg.use_cosine_loss else 0.0)
        loss = recon + out["commit_loss"]

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total["loss"] += float(loss.item())
        total["recon_mse"] += float(recon_mse.item())
        total["recon_cosloss"] += float(recon_cos.item())
        total["commit_loss"] += float(out["commit_loss"].item())

    n = max(len(loader), 1)
    return {k: v / n for k, v in total.items()}


def evaluate_codebook_metrics(
    indices_by_level: List[np.ndarray],
    codebook_size: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for l, idxs in enumerate(indices_by_level):
        cnt = np.bincount(idxs, minlength=codebook_size)
        p = cnt / max(cnt.sum(), 1)
        used = int((cnt > 0).sum())
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        perplexity = float(np.exp(entropy))
        dead = int((cnt == 0).sum())
        metrics[f"l{l+1}_usage_ratio"] = used / codebook_size
        metrics[f"l{l+1}_dead_codes"] = dead
        metrics[f"l{l+1}_entropy"] = entropy
        metrics[f"l{l+1}_perplexity"] = perplexity
    return metrics


@torch.no_grad()
def encode_all(
    model: RQVAE,
    x_np: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    x = torch.from_numpy(x_np.astype(np.float32))
    loader = DataLoader(x, batch_size=batch_size, shuffle=False)

    x_hat_all = []
    indices_levels: List[List[np.ndarray]] = []
    residual_norm_track: List[List[float]] = []

    for batch in loader:
        bx = batch.to(device)
        z = model.encoder(bx)
        zq, _, indices, residual_norms = model.rvq(z)
        x_hat = model.decoder(zq)

        x_hat_all.append(x_hat.cpu().numpy())
        if not indices_levels:
            indices_levels = [[] for _ in range(len(indices))]
        for i, idx in enumerate(indices):
            indices_levels[i].append(idx.cpu().numpy())
        residual_norm_track.append([float(v.item()) for v in residual_norms])

    x_hat_np = np.concatenate(x_hat_all, axis=0)
    indices_np = [np.concatenate(v, axis=0) for v in indices_levels]
    residual_curve = np.mean(np.asarray(residual_norm_track), axis=0).tolist()
    return {
        "x_hat": x_hat_np,
        "indices": indices_np,
        "residual_curve": residual_curve,
    }


def _fit_kmeans_centers(
    vectors: np.ndarray,
    codebook_size: int,
    seed: int,
    n_init: int,
    max_iter: int,
    verbose: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit KMeans centers for one RVQ level; repeats samples if n < K."""
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")

    n_samples, dim = vectors.shape
    if n_samples == 0:
        raise ValueError("cannot initialize codebook from empty vectors")

    if n_samples < codebook_size:
        rng = np.random.default_rng(seed)
        extra_idx = rng.choice(n_samples, size=codebook_size - n_samples, replace=True)
        centers = np.concatenate([vectors, vectors[extra_idx]], axis=0).astype(np.float32)
        counts = np.ones(codebook_size, dtype=np.float32)
        return centers, counts, 0.0

    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=codebook_size,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
    )
    labels = kmeans.fit_predict(vectors)
    centers = kmeans.cluster_centers_.astype(np.float32)
    counts = np.bincount(labels, minlength=codebook_size).astype(np.float32)
    counts[counts <= 0] = 1.0
    return centers, counts, float(kmeans.inertia_)


@torch.no_grad()
def initialize_rvq_codebook_with_kmeans(
    model: RQVAE,
    x_np: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
    max_samples: int,
    n_init: int,
    max_iter: int,
    verbose: int,
) -> Dict[str, Any]:
    """Initialize each residual codebook with KMeans over encoder latents/residuals."""
    model.eval()
    x = torch.from_numpy(x_np.astype(np.float32))
    loader = DataLoader(x, batch_size=batch_size, shuffle=False)

    z_chunks = []
    for batch in loader:
        z_chunks.append(model.encoder(batch.to(device)).detach().cpu().numpy())
    z_np = np.concatenate(z_chunks, axis=0).astype(np.float32)

    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(z_np) > max_samples:
        fit_idx = rng.choice(len(z_np), size=max_samples, replace=False)
        residual = z_np[fit_idx].copy()
    else:
        residual = z_np.copy()

    stats: Dict[str, Any] = {
        "method": "kmeans",
        "fit_samples": int(len(residual)),
        "total_samples": int(len(z_np)),
        "levels": [],
    }

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    level_iter = range(model.rvq.num_codebooks)
    if tqdm is not None:
        level_iter = tqdm(level_iter, desc="KMeans codebook init", total=model.rvq.num_codebooks)

    for level in level_iter:
        LOGGER.info(
            "KMeans init level %d/%d: samples=%d K=%d dim=%d",
            level + 1,
            model.rvq.num_codebooks,
            len(residual),
            model.rvq.codebook_size,
            residual.shape[1],
        )
        centers, counts, inertia = _fit_kmeans_centers(
            residual,
            model.rvq.codebook_size,
            seed=seed + level,
            n_init=n_init,
            max_iter=max_iter,
            verbose=verbose,
        )

        centers_t = torch.from_numpy(centers).to(device=device, dtype=model.rvq.embed.dtype)
        counts_t = torch.from_numpy(counts).to(device=device, dtype=model.rvq.cluster_size.dtype)
        model.rvq.embed[level].copy_(centers_t)
        model.rvq.cluster_size[level].copy_(counts_t)
        model.rvq.embed_avg[level].copy_(centers_t * counts_t.unsqueeze(1))

        # Quantize residuals with initialized centers before fitting the next residual level.
        dist = (
            np.sum(residual * residual, axis=1, keepdims=True)
            - 2.0 * residual @ centers.T
            + np.sum(centers * centers, axis=1)
        )
        labels = np.argmin(dist, axis=1)
        residual = residual - centers[labels]

        stats["levels"].append(
            {
                "level": level + 1,
                "inertia": inertia,
                "used_codes": int((counts > 0).sum()),
                "min_count": float(counts.min()),
                "max_count": float(counts.max()),
                "residual_norm_after": float(np.linalg.norm(residual, axis=1).mean()),
            }
        )

    return stats


def parse_json_objects(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    decoder = json.JSONDecoder()
    objs = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(text, i)
        objs.append(obj)
        i = j
    return objs


def normalize_criteria(criteria):
    return {str(k): v for k, v in (criteria or {}).items()}

def build_embedding_text(title: str, description: str, criteria: dict) -> str:
    return "\n".join([
        f"title: {title}",
        f"description: {description}",
        f"criteria_0: {criteria.get('0', '')}",
        f"criteria_1: {criteria.get('1', '')}",
    ])

def extract_rubric_items(path: Path) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Extract rubric items from generated_rubrics_*.jsonl.

    Expected record shape:
      {"query": ..., "rubric_model": ..., "rubric": {"dimensions": [...]}}

    Failed generation records in this file have an "error" field and
    rubric=null; those are skipped.  The original judge-result format is still
    accepted as long as it has query + rubric.dimensions.
    """
    objs = parse_json_objects(path)
    ids: List[str] = []
    texts: List[str] = []
    items: List[Dict[str, Any]] = []
    rid = 1
    skipped_error = 0
    skipped_no_rubric = 0
    skipped_no_dimensions = 0
    model_counts: Counter = Counter()

    for rec_i, o in enumerate(objs, start=1):
        if not isinstance(o, dict):
            skipped_no_rubric += 1
            continue
        if o.get("error"):
            skipped_error += 1
            continue

        query = o.get("query", "")
        rubric = o.get("rubric") or {}
        if not isinstance(rubric, dict):
            skipped_no_rubric += 1
            continue

        dimensions = rubric.get("dimensions") or []
        if not dimensions:
            skipped_no_dimensions += 1
            continue

        rubric_model = o.get("rubric_model") or rubric.get("rubric_model") or ""
        model_counts[rubric_model] += 1
        country_code = o.get("country_code", "")
        row_number = o.get("row_number")
        ecom_search_pv = o.get("ecom_search_pv", "")

        for dim in dimensions:
            if not isinstance(dim, dict):
                continue
            dimension = (dim.get("name") or "").strip()
            if not dimension:
                continue
            for item in dim.get("rubrics", []):
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "")
                desc = item.get("description", "")
                criteria = normalize_criteria(item.get("criteria"))
                embedding_text = build_embedding_text(title, desc, criteria)
                sid = str(rid)
                ids.append(sid)
                texts.append(embedding_text)
                items.append({
                    "rubric_id": rid,
                    "source_query": query,
                    "country_code": country_code,
                    "row_number": row_number,
                    "ecom_search_pv": ecom_search_pv,
                    "rubric_model": rubric_model,
                    "source_record_index": rec_i,
                    "dimension": dimension,
                    "dimension_weight": dim.get("weight"),
                    "is_set_level": dim.get("is_set_level"),
                    "title": title,
                    "description": desc,
                    "criteria": criteria,
                    "points": item.get("points"),
                    "embedding_text": embedding_text,
                })
                rid += 1

    LOGGER.info(
        "Skipped records: error=%d no_rubric=%d no_dimensions=%d; source model counts=%s",
        skipped_error,
        skipped_no_rubric,
        skipped_no_dimensions,
        dict(model_counts),
    )
    if not texts:
        raise ValueError(f"No rubric items extracted from {path}")
    return ids, texts, items


def build_experiment_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    grid = []
    for k in args.codebook_sizes:
        for d in args.num_codebooks:
            for ld in args.latent_dims:
                for beta in args.beta_commit:
                    grid.append(
                        {
                            "codebook_size": k,
                            "num_codebooks": d,
                            "latent_dim": ld,
                            "beta_commit": beta,
                        }
                    )
    return grid


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ-VAE title embedding pipeline")
    parser.add_argument("--input_json", default="/root/judge/generated_rubrics_6.30query_new_sg.jsonl")
    parser.add_argument("--output_dir", default="/root/judge/rqvae_outputs_generated_rubrics")
    parser.add_argument("--model_dir", default="/root/qwen3_embedding")
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cosine_weight", type=float, default=0.1)
    parser.add_argument("--no_cosine_loss", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--codebook_sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--num_codebooks", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--latent_dims", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--beta_commit", type=float, nargs="+", default=[0.25])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--init_codebook", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--kmeans_init_max_samples", type=int, default=50000)
    parser.add_argument("--kmeans_init_n_init", type=int, default=10)
    parser.add_argument("--kmeans_init_max_iter", type=int, default=300)
    parser.add_argument("--kmeans_init_verbose", type=int, default=1)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    input_path = Path(args.input_json)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    ids, texts, extracted_items = extract_rubric_items(input_path)
    LOGGER.info("Loaded %d rubric items from %s", len(texts), input_path)
    write_json(out_root / "rubrics_extracted_no_points.json", extracted_items)

    x_np = encode_texts_with_sentence_transformer(args.model_dir, texts, args.batch_size)
    if args.normalize:
        x_np = x_np / np.clip(np.linalg.norm(x_np, axis=1, keepdims=True), 1e-12, None)

    np.save(out_root / "titles_embeddings.npy", x_np)
    write_json(
        out_root / "meta.json",
        {
            "input_json": str(input_path),
            "num_rubrics": len(texts),
            "embedding_shape": list(x_np.shape),
            "normalized": bool(args.normalize),
        },
    )

    dataset = EmbeddingDataset(x_np)
    n_val = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val

    grid = build_experiment_grid(args)
    all_results: List[Dict[str, Any]] = []

    for exp_idx, hp in enumerate(grid, start=1):
        for seed in args.seeds:
            set_seed(seed)
            exp_name = (
                f"exp{exp_idx:03d}_k{hp['codebook_size']}_d{hp['num_codebooks']}"
                f"_ld{hp['latent_dim']}_b{hp['beta_commit']}_s{seed}"
            )
            exp_dir = out_root / exp_name
            ensure_dir(exp_dir)
            LOGGER.info("Start %s", exp_name)

            if n_val > 0:
                train_ds, val_ds = random_split(
                    dataset,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(seed),
                )
            else:
                train_ds, val_ds = dataset, dataset

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

            device = torch.device(args.device)
            model = RQVAE(
                input_dim=x_np.shape[1],
                latent_dim=hp["latent_dim"],
                num_codebooks=hp["num_codebooks"],
                codebook_size=hp["codebook_size"],
                beta_commit=hp["beta_commit"],
                ema_decay=args.ema_decay,
            ).to(device)

            init_stats: Dict[str, Any] = {"method": "random"}
            if args.init_codebook == "kmeans":
                LOGGER.info(
                    "%s KMeans init codebook: K=%d D=%d latent_dim=%d max_samples=%d",
                    exp_name,
                    hp["codebook_size"],
                    hp["num_codebooks"],
                    hp["latent_dim"],
                    args.kmeans_init_max_samples,
                )
                init_stats = initialize_rvq_codebook_with_kmeans(
                    model=model,
                    x_np=x_np,
                    batch_size=args.batch_size,
                    device=device,
                    seed=seed,
                    max_samples=args.kmeans_init_max_samples,
                    n_init=args.kmeans_init_n_init,
                    max_iter=args.kmeans_init_max_iter,
                    verbose=args.kmeans_init_verbose,
                )
                write_json(exp_dir / "init_stats.json", init_stats)
                LOGGER.info("%s KMeans init done: %s", exp_name, init_stats["levels"])
            else:
                write_json(exp_dir / "init_stats.json", init_stats)

            cfg = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                cosine_weight=args.cosine_weight,
                use_cosine_loss=not args.no_cosine_loss,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

            history: List[Dict[str, float]] = []
            best_val = float("inf")
            best_state = None

            for epoch in range(1, cfg.epochs + 1):
                train_m = run_epoch(model, train_loader, optimizer, cfg, device)
                val_m = run_epoch(model, val_loader, None, cfg, device)
                row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
                history.append(row)
                if val_m["loss"] < best_val:
                    best_val = val_m["loss"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
                    LOGGER.info("%s epoch=%d train_loss=%.6f val_loss=%.6f", exp_name, epoch, train_m["loss"], val_m["loss"])

            if best_state is not None:
                model.load_state_dict(best_state)
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            write_json(exp_dir / "history.json", history)

            enc = encode_all(model, x_np, args.batch_size, device)
            x_hat = enc["x_hat"]
            idx_levels = enc["indices"]

            recon_mse = float(np.mean((x_hat - x_np) ** 2))
            x_norm = np.linalg.norm(x_np, axis=1) + 1e-12
            rel_l2 = float(np.mean(np.linalg.norm(x_hat - x_np, axis=1) / x_norm))
            cos = float(np.mean(np.sum(x_hat * x_np, axis=1) / (np.linalg.norm(x_hat, axis=1) * np.linalg.norm(x_np, axis=1) + 1e-12)))

            code_m = evaluate_codebook_metrics(idx_levels, hp["codebook_size"])
            joint_codes = list(zip(*[v.tolist() for v in idx_levels]))
            joint_counter = Counter(joint_codes)
            small_cluster_ratio = float(np.mean([1 if c < 3 else 0 for c in joint_counter.values()])) if joint_counter else 0.0

            title_code_rows = []
            for i, sid in enumerate(ids):
                base = extracted_items[i]
                row = {
                    "rubric_id": int(base["rubric_id"]),
                    "source_query": base["source_query"],
                    "country_code": base.get("country_code", ""),
                    "row_number": base.get("row_number"),
                    "ecom_search_pv": base.get("ecom_search_pv", ""),
                    "rubric_model": base.get("rubric_model", ""),
                    "source_record_index": base.get("source_record_index"),
                    "dimension": base["dimension"],
                    "dimension_weight": base.get("dimension_weight"),
                    "is_set_level": base.get("is_set_level"),
                    "title": base["title"],
                    "description": base["description"],
                    "criteria": base["criteria"],
                    "points": base.get("points"),
                }
                for l, arr in enumerate(idx_levels, start=1):
                    row[f"code_l{l}"] = int(arr[i])
                title_code_rows.append(row)

            with (exp_dir / "rubric_codes.jsonl").open("w", encoding="utf-8") as f:
                for r in title_code_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            l1_dist = Counter(idx_levels[0].tolist()) if idx_levels else Counter()
            write_json(exp_dir / "l1_distribution.json", {str(k): v for k, v in sorted(l1_dist.items())})
            write_json(
                exp_dir / "joint_distribution_top200.json",
                [
                    {"code": list(k), "count": v}
                    for k, v in joint_counter.most_common(200)
                ],
            )

            summary = {
                "exp_name": exp_name,
                "seed": seed,
                **hp,
                "recon_mse": recon_mse,
                "recon_cos": cos,
                "relative_l2_error": rel_l2,
                "init_codebook": args.init_codebook,
                "init_stats": init_stats,
                "small_cluster_ratio_lt3": small_cluster_ratio,
                "joint_code_count": len(joint_counter),
                "residual_curve": enc["residual_curve"],
                **code_m,
            }
            write_json(exp_dir / "summary.json", summary)
            all_results.append(summary)

    all_results_sorted = sorted(all_results, key=lambda x: (x["recon_mse"], -x.get("l1_usage_ratio", 0.0)))
    write_json(out_root / "all_experiments_summary.json", all_results_sorted)

    LOGGER.info("Done. %d experiment runs finished.", len(all_results))


if __name__ == "__main__":
    main()
