import scipy.stats
from typing import Dict, List


def gen_loss_scenario(freq_model:str, freq_param: Dict[str, float], sev_model: str, sev_param: Dict[str, float], exposure: float, num_scen: int) -> Dict[str, List[float]]:
  freq = getattr(scipy.stats, freq_model)(**freq_param)
  sev = getattr(scipy.stats, sev_model)(**sev_param)
  num_claims_sample = freq.rvs(num_scen).tolist()
  total_loss_sample = [sev.rvs(n).sum() for n in num_claims_sample]
  return dict(num_claims=num_claims_sample, total_loss=total_loss_sample)
  

# if __name__ == '__main__':
#   freq_model = 'poisson'
#   freq_param = {'mu': 0.6}
#   sev_model = 'gamma'
#   sev_param = {'a': 1., 'loc': 0, 'scale': 1000}
#   gen_loss_scenario(freq_model, freq_param, sev_model, sev_param, 1., 10)