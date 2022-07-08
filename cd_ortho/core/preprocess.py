"""Preprocess module, handles data preprocessing inside A Dataset class"""
from typing import Dict


class UniversalPreprocessor:

    def __init__(self, input_fields):

        self._input_fields = input_fields

    def __call__(self, input: Dict, *args, **kwargs):

        ...
