import sys
import os
import glob
import importlib

import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from datetime import datetime

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=4"

import jax
jax.config.update('jax_enable_x64', True)

import jax.random
import jax.numpy as jnp

import discovery as ds
import discovery.models.nanograv as ds_nanograv
import discovery.samplers.numpyro as ds_numpyro

from discovery import prior
from discovery.params import Params, make_layout

import numpyro
from numpyro import infer, distributions as dist

import arviz

psr = ds.Pulsar.read_feather('/DATA/CARINA_4/mit227/subband_PA_data/J0835-4510.PA.head_derot.minusSBmean.feather', json_metadata=False)
psr.dist0 = 0.280 # kpc

m_aldm = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr, tnequad=True, log_efac=True),
                         ds.makegp_ecorr(psr),
                         ds.makegp_erm(psr),
                         ds.makegp_fourier(psr, ds.powerlaw, components=30, fourierbasis=ds.fourierbasis_dm, name='rmgp'),
                         ds.makegp_aldm_psr(psr,-23.0)
                         ])

logL = m_aldm.logL

# static layout shared by every Params built inside the model
layout, size = make_layout(logL.params)

# flat uniform bounds, one entry per (scalar) parameter
bounds = [prior.getprior_uniform(p) for p in logL.params]
print(bounds)
lo = jnp.array([b[0] for b in bounds])
hi = jnp.array([b[1] for b in bounds])

def numpyro_model():
    raw = numpyro.sample('raw', dist.Uniform(lo, hi))   # one (P,) array site
    numpyro.factor('logl', logL(Params(raw, layout)))

mcmc = infer.MCMC(infer.NUTS(numpyro_model), num_chains=4,
                     num_warmup=100, num_samples=100)
jax.debug.print('Starting warmup')
# mcmc.warmup(jax.random.PRNGKey(42))
# jax.debug.print('Warmup finished')
# jax.debug.print('Starting run')
# mcmc.run(mcmc.post_warmup_state.rng_key)
mcmc.run(jax.random.PRNGKey(42))
jax.debug.print('Run finished')
# mcmc.run([jax.random.PRNGKey(43+i) for i in range(n_chains)])

samples = mcmc.get_samples()


booking_path = os.environ.get("BOOK")
fit_write_path = f'{booking_path}/fit_results'
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S") # Format as YYYYMMDD-HHMMSS (e.g., 20261027-114200)

np.savez(f"{fit_write_path}/{psr.name}_mcmc_samples_{timestamp}.npz", **samples)

# idata = arviz.from_numpyro(mcmc)
# idata.to_netcdf(f"{fit_write_path}/{psr.name}_mcmc_inference_data_{timestamp}.nc")


raw_samples = samples['raw']

p_fit = {}
for i, name in enumerate(logL.params):
    col = raw_samples[:, i]
    print(f'{name:30s} mean={col.mean():+.3f}  std={col.std():.3f}  bounds={bounds[i]}')
    p_fit[name] = col.mean()

print(p_fit)
print(f'{logL(p_fit):e}')

figure = corner.corner(
    np.array(raw_samples),
    labels=logL.params
    )

plt.savefig(f"{fit_write_path}/{psr.name}_mcmc_samples_{timestamp}.pdf")
plt.show()

