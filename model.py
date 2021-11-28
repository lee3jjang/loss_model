from typing import Dict
from pydantic import BaseModel


class FreqSevModelConfig(BaseModel):
  freq_model: str
  freq_param: Dict[str, float]
  sev_model: str
  sev_param: Dict[str, float]
  exposure: float
  num_scen: int