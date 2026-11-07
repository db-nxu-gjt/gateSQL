# -*- coding: utf-8 -*-
"""llm_client.py — OpenAI 兼容客户端（生成候选 / verbalized 置信 / judge）。

配置（环境变量优先）：
    GATE_LLM_BASE_URL  如 https://api.deepseek.com/v1 或本地 vLLM http://host:8000/v1
    GATE_LLM_API_KEY   API key
    GATE_LLM_MODEL     默认模型名

支持：
  - chat.generate(): n 路采样（nucleus / 低温近似 beam）
  - chat.generate_with_logprobs(): 返回 content token 级 logprobs（B3/B4 白盒基线；
    需 vLLM/支持 logprobs 的端点 + 开源模型）
  - 断点续跑由各 run_* 脚本基于输出文件自行实现
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests


class LLMClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: int = 120):
        self.base_url = (base_url or os.environ.get("GATE_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("GATE_LLM_API_KEY", "EMPTY")
        self.model = model or os.environ.get("GATE_LLM_MODEL", "")
        self.timeout = timeout
        if not self.base_url:
            raise RuntimeError("GATE_LLM_BASE_URL 未设置")

    # ------------------------------------------------------------------ #
    def _post(self, payload: dict, max_retry: int = 5) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        for attempt in range(max_retry):
            try:
                r = requests.post(url, headers=headers, json=payload,
                                  timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt * 2)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code != 429 and code < 500:
                    raise  # 客户端错误（如参数不支持）重试无意义
                if attempt == max_retry - 1:
                    raise
                time.sleep(2 ** attempt * 2)
            except (requests.RequestException, ValueError):
                if attempt == max_retry - 1:
                    raise
                time.sleep(2 ** attempt * 2)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0,
                 top_p: float = 1.0, max_tokens: int = 512,
                 system: str = "", model: Optional[str] = None,
                 stop: Optional[list] = None,
                 extra: Optional[dict] = None) -> list[str]:
        """返回 n 条补全文本。beam10 的 API 近似：temperature≈0.2。

        兼容 Ollama 等静默忽略 n 的端点：一次调用只回 1 条时逐条补齐
        （temperature>0 下每次调用都是独立采样，多样性协议不受影响）。
        """
        payload = {
            "model": model or self.model, "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]),
            "n": n, "temperature": temperature, "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if extra:
            payload.update(extra)
        texts: list[str] = []
        while len(texts) < n:
            try:
                data = self._post(payload)
            except requests.HTTPError:
                if payload.get("n", 1) != 1:
                    payload["n"] = 1  # 端点仅支持 n=1（如 DeepSeek）→ 逐条采样
                    continue
                raise
            texts.extend(c["message"]["content"] or ""
                         for c in data.get("choices", []))
            if not data.get("choices"):
                break  # 防御：空响应避免死循环
        return texts[:n]

    def generate_with_logprobs(self, prompt: str, n: int = 1,
                               temperature: float = 0.0, top_p: float = 1.0,
                               max_tokens: int = 512, top_logprobs: int = 20,
                               system: str = "",
                               model: Optional[str] = None,
                               extra: Optional[dict] = None) -> list[dict]:
        """返回 [{"text":..., "tokens":[...], "token_logprobs":[...]}]。

        仅 content token 的 logprob（chat.completions logprobs 协议）。
        """
        payload = {
            "model": model or self.model, "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]),
            "n": n, "temperature": temperature, "top_p": top_p,
            "max_tokens": max_tokens, "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        if extra:
            payload.update(extra)
        out: list[dict] = []
        while len(out) < n:  # 同 generate：端点不支持 n 时（DeepSeek 400）逐条补齐
            try:
                data = self._post(payload)
            except requests.HTTPError:
                if payload.get("n", 1) != 1:
                    payload["n"] = 1
                    continue
                raise
            for c in data.get("choices", []):
                content = (c.get("logprobs") or {}).get("content") or []
                out.append({
                    "text": c["message"]["content"] or "",
                    "tokens": [t["token"] for t in content],
                    "token_logprobs": [t["logprob"] for t in content],
                })
            if not data.get("choices"):
                break
        return out[:n]
