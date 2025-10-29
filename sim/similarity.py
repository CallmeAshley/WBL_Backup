# -*- coding: utf-8 -*-
"""
전역 question 유사도 계산 스크립트 (Upstage Embeddings 기반)
- CLI 없이 상단 CONFIG만 바꾸고 바로 실행하세요.
- 출력: topk_neighbors.csv, similar_pairs.csv, summary.json
"""

import os
import sys
import json
import glob
import time
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

# ====== (선택) FAISS가 있으면 고속, 없으면 sklearn 대체 ======
_USE_FAISS = False
try:
    import faiss  # type: ignore
    _USE_FAISS = True
except Exception:
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except Exception:
        raise RuntimeError("faiss 또는 scikit-learn 중 하나가 필요합니다. pip install faiss-cpu 또는 scikit-learn")



# =========================
# CONFIG: 여기만 바꿔서 실행
# =========================
@dataclass
class Config:
    data_dir: str = "/mnt/c/Users/Flitto/Documents/WBL/processed/total"                # jsonl들이 들어있는 상위 폴더
    file_pattern: str = "**/*.jsonl"        # 재귀 패턴
    field_name: str = "question"            # 객체 내 필드명
    snippet_len: int = 80                   # 미리보기 길이
    dedupe_exact: bool = True               # 완전 일치 문장 중복 제거
    model: str = "embedding-passage"        # Upstage 권장: embedding-passage
    batch_size: int = 100                   # Upstage: 요청당 최대 100
    k: int = 5                              # Top-K (자기 자신 제외 후 K개)
    threshold: float = 0.90                 # 임계값 필터 (cosine >= T)
    backend: str = "faiss"                  # "faiss" 또는 "sklearn"
    cache_path: str = ".embed_cache.pkl"    # 로컬 캐시 파일
    out_topk_csv: str = "/mnt/c/Users/Flitto/Documents/WBL/sim/topk_neighbors."
    out_pairs_csv: str = "/mnt/c/Users/Flitto/Documents/WBL/sim/similar_pairs.csv"
    out_summary_json: str = "/mnt/c/Users/Flitto/Documents/WBL/sim/summary.json"


CFG = Config()

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def read_jsonl_files(data_dir: str, pattern: str, field: str) -> List[Tuple[str, str]]:
    files = glob.glob(os.path.join(data_dir, pattern), recursive=True)
    rows: List[Tuple[str, str]] = []
    cnt = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # 잘못된 JSON은 스킵
                    continue
                if field not in obj:
                    continue
                text = str(obj[field]).strip()
                if not text:
                    continue
                qid = f"G{cnt:05d}"
                rows.append((qid, text))
                cnt += 1
    return rows


def dedupe_texts(qid_texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    for qid, txt in qid_texts:
        if txt in seen:
            continue
        seen.add(txt)
        out.append((qid, txt))
    return out

from api_set import client


class UpstageEmbedder:
    def __init__(self, model: str, batch_size: int = 100, cache_path: str = ".embed_cache.pkl", client=None):
        self.model = model
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.client = client
        if not self.client.api_key:
            raise RuntimeError("환경변수 UPSTAGE_API_KEY가 설정되어 있지 않습니다.")

        # 캐시 로드
        self.cache: Dict[str, List[float]] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    def save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        to_query_idx: List[int] = []
        to_query_texts: List[str] = []

        # 캐시 조회
        cached_vectors: Dict[int, np.ndarray] = {}
        for i, t in enumerate(texts):
            key = f"{self.model}:{sha1(t)}"
            if key in self.cache:
                cached_vectors[i] = np.array(self.cache[key], dtype=np.float32)
            else:
                to_query_idx.append(i)
                to_query_texts.append(t)

        # 배치 호출
        for s in range(0, len(to_query_texts), self.batch_size):
            batch = to_query_texts[s: s + self.batch_size]
            # 재시도 간단 구현
            for attempt in range(5):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    data = resp.data
                    for j, item in enumerate(data):
                        vec = np.array(item.embedding, dtype=np.float32)
                        idx = to_query_idx[s + j]
                        cached_vectors[idx] = vec
                        # 캐시 기록
                        key = f"{self.model}:{sha1(batch[j])}"
                        self.cache[key] = item.embedding
                    break
                except Exception as e:
                    sleep_s = 1.5 * (attempt + 1)
                    time.sleep(sleep_s)
                    if attempt == 4:
                        raise e  # 5회 실패 시 에러

        # 정렬 결합
        dim = len(next(iter(cached_vectors.values())))
        E = np.zeros((len(texts), dim), dtype=np.float32)
        for i in range(len(texts)):
            E[i] = cached_vectors[i]

        # 안전 재정규화
        E = normalize_rows(E)

        # 캐시 저장
        self.save_cache()

        return E

def knn_search(E: np.ndarray, k: int, backend: str = "faiss") -> Tuple[np.ndarray, np.ndarray]:
    """
    자기 자신 포함 K 이웃을 반환한 뒤, 호출자가 self를 제거/보정하도록 함.
    반환:
      I: (N, k) 이웃 인덱스
      S: (N, k) 유사도 (cosine)
    """
    N, d = E.shape
    if backend == "faiss" and _USE_FAISS:
        index = faiss.IndexFlatIP(d)  # inner product == cosine (정규화 가정)
        index.add(E.astype(np.float32))
        S, I = index.search(E.astype(np.float32), k)
        # FAISS는 유사도(S) 내림차순
        return I, S
    else:
        # sklearn: metric='cosine'은 "거리"를 반환 → similarity = 1 - distance
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(E)
        distances, indices = nn.kneighbors(E, n_neighbors=k, return_distance=True)
        sims = 1.0 - distances
        return indices, sims


def build_topk_rows(
    qids: List[str],
    texts: List[str],
    I: np.ndarray,
    S: np.ndarray,
    snippet_len: int,
    k: int
) -> List[Dict[str, Any]]:
    N = len(qids)
    rows: List[Dict[str, Any]] = []
    for i in range(N):
        # 자기 자신 제거
        neigh_idx = I[i]
        neigh_sim = S[i]

        pairs = [(j, float(s)) for j, s in zip(neigh_idx, neigh_sim) if j != i]
        pairs = pairs[:k] 
        for j, score in pairs:
            rows.append({
                "qid": qids[i],
                "question_snippet": texts[i][:snippet_len].replace("\n", " "),
                "nn_qid": qids[j],
                "nn_snippet": texts[j][:snippet_len].replace("\n", " "),
                "cosine": round(score, 6),
            })
    return rows


def filter_pairs_by_threshold(
    topk_rows: List[Dict[str, Any]],
    threshold: float
) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for r in topk_rows:
        if r["cosine"] < threshold:
            continue
        a, b = r["qid"], r["nn_qid"]
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "qid_left": key[0],
            "qid_right": key[1],
            "left_snippet": r["question_snippet"] if key[0] == a else r["nn_snippet"],
            "right_snippet": r["nn_snippet"] if key[0] == a else r["question_snippet"],
            "cosine": r["cosine"],
        })
    out.sort(key=lambda x: x["cosine"], reverse=True)
    return out


def write_excel(path: str, rows: list[dict], sheet_name: str = "Sheet1"):
    if not rows:
        print(f"저장할 데이터가 없습니다: {path}")
        return
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, sheet_name=sheet_name)
    print(f"Excel 저장 완료: {path} (rows={len(df)})")


def write_summary(path: str, E: np.ndarray, topk_rows: List[Dict[str, Any]], model: str):
    sims = [r["cosine"] for r in topk_rows]
    summary = {
        "count": int(E.shape[0]),
        "pairs": int(len(topk_rows)),
        "mean": float(np.mean(sims)) if sims else None,
        "p90": float(np.percentile(sims, 90)) if sims else None,
        "p95": float(np.percentile(sims, 95)) if sims else None,
        "max": float(np.max(sims)) if sims else None,
        "model": model,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    cfg = CFG
    qid_texts = read_jsonl_files(cfg.data_dir, cfg.file_pattern, cfg.field_name)
    if not qid_texts:
        print("읽을 question이 없습니다. 경로/필드/패턴을 확인하세요.")
        sys.exit(0)

    if cfg.dedupe_exact:
        qid_texts = dedupe_texts(qid_texts)

    qids = [qt[0] for qt in qid_texts]
    texts = [qt[1] for qt in qid_texts]

    print(f"총 문장 수: {len(texts)}")
    
    embedder = UpstageEmbedder(model=cfg.model, batch_size=cfg.batch_size, cache_path=cfg.cache_path, client=client)
    E = embedder.embed(texts) 
    
    K_query = cfg.k + 1
    backend = cfg.backend if cfg.backend in ("faiss", "sklearn") else ("faiss" if _USE_FAISS else "sklearn")
    I, S = knn_search(E, k=K_query, backend=backend)

    topk_rows = build_topk_rows(qids, texts, I, S, cfg.snippet_len, cfg.k)

    pair_rows = filter_pairs_by_threshold(topk_rows, cfg.threshold)

    write_excel("topk_neighbors.xlsx", topk_rows, sheet_name="TopK")
    write_excel("similar_pairs.xlsx", pair_rows, sheet_name="Pairs")
    write_summary(cfg.out_summary_json, E, topk_rows, cfg.model)

    print(f"저장 완료:")
    print(f" - Top-K: {cfg.out_topk_csv} (rows={len(topk_rows)})")
    print(f" - Pairs (≥{cfg.threshold}): {cfg.out_pairs_csv} (rows={len(pair_rows)})")
    print(f" - Summary: {cfg.out_summary_json}")


if __name__ == "__main__":
    main()
