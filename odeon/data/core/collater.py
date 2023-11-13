from typing import Dict


class Collater:

    def __init__(self, input_fields: Dict):

        self.input_fields = input_fields

    def __call__(self, input: Dict, *args, **kwargs):

        ...
