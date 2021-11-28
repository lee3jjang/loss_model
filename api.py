from fastapi import FastAPI
from starlette.responses import JSONResponse

from core import gen_loss_scenario
from model import FreqSevModelConfig


app = FastAPI()

@app.post('/scen')
def get_loss_scenario(config: FreqSevModelConfig):
  return gen_loss_scenario(config.freq_model, config.freq_param,
    config.sev_model, config.sev_param, config.exposure, config.num_scen)