import numpy as np

processes = 4


def make_exponential_anneal_protocol(start_temp=1, stop_temp=1e-5, warmup = 500, half_life = 300):
  protocol = np.concatenate((
    np.ones(warmup)*start_temp,
    np.logspace(np.log(start_temp), np.log(stop_temp), int(np.ceil((np.log(start_temp)-np.log(stop_temp))/np.log(2)*half_life + 1)), base=np.e)
  ))
  return list(protocol)

anneal_protocol = make_exponential_anneal_protocol(half_life = 300)
train_kwargs_MG = {
  'anneal_protocol': anneal_protocol,
  'warmup_iters': len(anneal_protocol)
}
