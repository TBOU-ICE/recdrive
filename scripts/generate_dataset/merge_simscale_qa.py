"""
Merge per-shard SimScale QA jsonl into the *_all.jsonl the meta points to, with
allowlist-based quality filtering + token dedup.

This is robust to:
  * arbitrary shard counts (globs simscale_<ds>_<kind>_shard*of*.jsonl)
  * stale / mixed-provenance shards (e.g. an earlier pre-QC run left behind
    out-of-allowlist records that RESUME=1 did not overwrite) -> they are dropped
  * duplicate tokens across shards -> first occurrence kept

Run:
  SIMSCALE_ROOT=/workspace/datasets/simscale/20260709 ROUND=0 \
  python scripts/generate_dataset/merge_simscale_qa.py

Env:
  SIMSCALE_ROOT       (default /workspace/datasets/simscale/20260709)
  ROUND               0 | 1              (selects synthetic_reaction_pdm_v1.0-<ROUND>)
  SIMSCALE_QA_OUT_DIR (default <root>/simscale_vlm_qa/<dataset>)
  QUALITY_FILTER      1 (default) | 0    apply allowlist filter
  PDMS_THRESHOLD      >0 to select tokens with PDMS>=threshold from scores CSV
  SIMSCALE_ALLOWLIST  explicit allowlist path (tsv col2 or 1-col)
  KINDS               comma list, default "traj,qa"
"""

import os
import csv
import json
import glob
from pathlib import Path


def load_allowlist(simscale_root: Path, ds: str, pdms_threshold: float):
    qc_dir = simscale_root / f"quality_filter_{ds}"
    override = os.environ.get("SIMSCALE_ALLOWLIST", "")
    if override:
        src = Path(override)
        toks = set()
        for line in open(src):
            line = line.strip()
            if line:
                p = line.split("\t")
                toks.add(p[1] if len(p) >= 2 else p[0])
        print(f"[merge][qc] allowlist(override)={src} tokens={len(toks)}")
        return toks
    if pdms_threshold > 0:
        csv_path = qc_dir / "simscale_target_quality_scores.csv"
        if csv_path.is_file():
            toks = set()
            for row in csv.DictReader(open(csv_path)):
                try:
                    if float(row["PDMS"]) >= pdms_threshold:
                        toks.add(row["token"])
                except (KeyError, ValueError):
                    pass
            print(f"[merge][qc] PDMS>={pdms_threshold} tokens={len(toks)}")
            return toks
        print(f"[merge][qc] PDMS_THRESHOLD set but {csv_path} missing; fall back to allowlist")
    for name in ("allowlist_tokens.txt", "allowlist_log_token.tsv"):
        src = qc_dir / name
        if src.is_file():
            toks = set()
            for line in open(src):
                line = line.strip()
                if line:
                    p = line.split("\t")
                    toks.add(p[1] if len(p) >= 2 else p[0])
            print(f"[merge][qc] allowlist={name} tokens={len(toks)}")
            return toks
    print(f"[merge][qc] no allowlist under {qc_dir}; QC DISABLED")
    return None


def merge_kind(out_dir: Path, ds: str, kind: str, allow):
    parts = sorted(glob.glob(str(out_dir / f"simscale_{ds}_{kind}_shard*of*.jsonl")))
    out = out_dir / f"simscale_{ds}_{kind}_all.jsonl"
    if not parts:
        print(f"[merge] {kind}: no shards found; skip")
        return
    seen = set()
    kept = dropped_qc = dup = 0
    with open(out, "w", encoding="utf-8") as w:
        for fp in parts:
            for line in open(fp):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    tok = json.loads(line)["token"]
                except Exception:
                    continue
                if allow is not None and tok not in allow:
                    dropped_qc += 1
                    continue
                if tok in seen:
                    dup += 1
                    continue
                seen.add(tok)
                w.write(line + "\n")
                kept += 1
    print(f"[merge] {kind}: shards={len(parts)} kept={kept} dropped_qc={dropped_qc} dup={dup} -> {out}")


def main():
    simscale_root = Path(os.environ.get("SIMSCALE_ROOT", "/workspace/datasets/simscale/20260709"))
    round_id = os.environ.get("ROUND", "0")
    ds = f"synthetic_reaction_pdm_v1.0-{round_id}"
    out_dir = Path(os.environ.get("SIMSCALE_QA_OUT_DIR", str(simscale_root / "simscale_vlm_qa" / ds)))
    quality_filter = os.environ.get("QUALITY_FILTER", "1") == "1"
    pdms_threshold = float(os.environ.get("PDMS_THRESHOLD", "0"))
    kinds = [k.strip() for k in os.environ.get("KINDS", "traj,qa").split(",") if k.strip()]

    allow = load_allowlist(simscale_root, ds, pdms_threshold) if quality_filter else None
    print(f"[merge] dataset={ds} out_dir={out_dir} quality_filter={quality_filter} "
          f"allow={'None' if allow is None else len(allow)}")
    for kind in kinds:
        merge_kind(out_dir, ds, kind, allow)


if __name__ == "__main__":
    main()
