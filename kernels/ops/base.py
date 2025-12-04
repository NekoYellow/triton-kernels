from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type, Tuple, Any
import argparse
import torch


@dataclass
class OpConfig:
    name: str
    description: str = ""


class BaseOp:

    config: OpConfig

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        pass

    @staticmethod
    def make_inputs(args: argparse.Namespace) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @staticmethod
    def torch_impl(*tensors: torch.Tensor) -> Any:
        raise NotImplementedError

    @staticmethod
    def triton_impl(*tensors: torch.Tensor) -> Any:
        raise NotImplementedError


OP_REGISTRY: Dict[str, Type[BaseOp]] = {}


def register_op(cls: Type[BaseOp]) -> Type[BaseOp]:
    name = cls.config.name
    if name in OP_REGISTRY:
        raise ValueError(f"Op {name} already registered")
    OP_REGISTRY[name] = cls
    return cls
