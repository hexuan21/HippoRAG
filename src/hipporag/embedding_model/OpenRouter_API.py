from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

#### in debugging ####
class OpenRouter_Embed_Model(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        # Default model name
        self.embedding_model_name = embedding_model_name or "Qwen/Qwen3-Embedding-0.6B"
        if embedding_model_name is not None:
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}"
            )

        self._init_embedding_config()

        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}"
        )

        # Tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name,
            trust_remote_code=self.embedding_config.model_init_params.get("trust_remote_code", False),
        )
        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)

        # Try to infer embedding dim robustly
        self.embedding_dim = getattr(self.embedding_model.config, "hidden_size", None)
        if self.embedding_dim is None:
            # Fallbacks for some configs
            self.embedding_dim = getattr(self.embedding_model.config, "d_model", None)
        if self.embedding_dim is None:
            raise ValueError("Cannot infer embedding_dim from model config (hidden_size/d_model missing).")

        self.embedding_model.eval()

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": self.global_config.embedding_model_dtype,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,  # kept for parity; not used directly here
                "pooling": "mean",  # mean | cls
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling with attention mask.
        last_hidden_state: [B, L, H]
        attention_mask:    [B, L]
        """
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B, 1]
        return summed / counts

    def _format_with_instruction(self, texts: List[str], instruction: str) -> List[str]:
        """
        Optional lightweight instruction formatting.
        You can customize this if your Qwen embedding model expects a specific template.
        """
        if not instruction:
            return texts
        # Mirror the style used in your NVEmbed wrapper
        prefix = f"Instruct: {instruction}\nQuery: "
        return [prefix + t for t in texts]

    @torch.inference_mode()
    def _encode_batch(self, texts: List[str], max_length: int, pooling: str) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to the model device (works with device_map="auto")
        device = next(self.embedding_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.embedding_model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, L, H]

        if pooling == "cls":
            emb = last_hidden[:, 0, :]
        else:
            emb = self._mean_pool(last_hidden, inputs["attention_mask"])

        return emb

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        instruction = params.pop("instruction", "") or ""
        max_length = int(params.pop("max_length", 512))
        batch_size = int(params.pop("batch_size", 16))
        pooling = str(params.pop("pooling", "mean")).lower()

        # Apply instruction formatting if provided
        texts = self._format_with_instruction(texts, instruction)

        logger.debug(f"Calling {self.__class__.__name__} with: max_length={max_length}, batch_size={batch_size}, pooling={pooling}")

        if len(texts) <= batch_size:
            emb = self._encode_batch(texts, max_length=max_length, pooling=pooling)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            chunks = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                chunks.append(self._encode_batch(chunk, max_length=max_length, pooling=pooling))
                pbar.update(len(chunk))
            pbar.close()
            emb = torch.cat(chunks, dim=0)

        # To CPU numpy
        emb = emb.detach().cpu().numpy()

        # Optional L2 normalize
        if self.embedding_config.norm:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            emb = emb / norms

        return emb
