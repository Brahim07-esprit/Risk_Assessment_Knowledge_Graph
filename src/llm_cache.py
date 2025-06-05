import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMCache:
    def __init__(self, cache_dir: str = ".llm_cache"):
        self.cache_dir = cache_dir
        self.enabled = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
        self.ttl_hours = int(os.getenv("LLM_CACHE_TTL_HOURS", "24"))

        if self.enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.memory_cache = {}

    def get_cache_key(
            self,
            prompt: str,
            model: str,
            temperature: float) -> str:
        cache_input = f"{model}:{temperature}:{prompt}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def get(
            self,
            prompt: str,
            model: str,
            temperature: float) -> Optional[Dict]:
        if not self.enabled:
            return None

        cache_key = self.get_cache_key(prompt, model, temperature)

        if cache_key in self.memory_cache:
            cached = self.memory_cache[cache_key]
            if self._is_valid(cached):
                logger.info(f"Cache hit (memory): {cache_key[:8]}...")
                return cached["response"]

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)

                if self._is_valid(cached):
                    self.memory_cache[cache_key] = cached
                    logger.info(f"Cache hit (disk): {cache_key[:8]}...")
                    return cached["response"]
                else:
                    os.remove(cache_file)

            except Exception as e:
                logger.error(f"Error reading cache: {e}")

        return None

    def set(
            self,
            prompt: str,
            model: str,
            temperature: float,
            response: Dict) -> None:
        if not self.enabled:
            return

        cache_key = self.get_cache_key(prompt, model, temperature)

        cached_data = {
            "prompt": prompt[:200],
            "model": model,
            "temperature": temperature,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "ttl_hours": self.ttl_hours,
        }

        self.memory_cache[cache_key] = cached_data

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(cached_data, f)
            logger.info(f"Cached response: {cache_key[:8]}...")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _is_valid(self, cached: Dict) -> bool:
        timestamp = datetime.fromisoformat(cached["timestamp"])
        ttl = timedelta(hours=cached.get("ttl_hours", self.ttl_hours))
        return datetime.now() - timestamp < ttl

    def clear_expired(self) -> int:
        if not self.enabled:
            return 0

        cleared = 0

        expired_keys = []
        for key, cached in self.memory_cache.items():
            if not self._is_valid(cached):
                expired_keys.append(key)

        for key in expired_keys:
            del self.memory_cache[key]
            cleared += 1

        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        with open(filepath, "r") as f:
                            cached = json.load(f)
                        if not self._is_valid(cached):
                            os.remove(filepath)
                            cleared += 1
                    except BaseException:
                        os.remove(filepath)
                        cleared += 1

        logger.info(f"Cleared {cleared} expired cache entries")
        return cleared

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "enabled": self.enabled,
            "memory_entries": len(self.memory_cache),
            "disk_entries": 0,
            "total_size_mb": 0,
        }

        if self.enabled and os.path.exists(self.cache_dir):
            disk_files = [
                f for f in os.listdir(
                    self.cache_dir) if f.endswith(".json")]
            stats["disk_entries"] = len(disk_files)

            total_size = sum(
                os.path.getsize(
                    os.path.join(
                        self.cache_dir,
                        f)) for f in disk_files)
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        return stats


class CachedLLMClient:

    def __init__(self, client, cache: LLMCache):
        self.client = client
        self.cache = cache

    def generate_with_cache(
        self, prompt: str, model: str, temperature: float, provider: str = "openai"
    ) -> Tuple[Optional[Dict], bool]:
        cached_response = self.cache.get(prompt, model, temperature)
        if cached_response:
            return cached_response, True

        try:
            if provider == "openai":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a risk assessment expert. Always return valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )

                result = {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "model": model,
                    "cached": False,
                }

            elif provider == "anthropic":
                response = self.client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=4000,
                )

                result = {
                    "content": response.content[0].text,
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens
                        + response.usage.output_tokens,
                    },
                    "model": model,
                    "cached": False,
                }

            self.cache.set(prompt, model, temperature, result)

            return result, False

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return None, False
