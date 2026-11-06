import sys
import os
import glob
import importlib
import argparse
import csv

import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from datetime import datetime

#os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=4"

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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--psrname",required=True,help='name of pulsar to run procedure on')
    # parser.add_argument("-m","--logmass",required=False,help='log10(mass/eV) to obtain bounds for')
    # parser.add_argument("-f","--freqmult",required=False,help='test the mass at this number of times 1/(duration of timeseries)')
    return parser.parse_args()

args = get_arguments()
psr_name = args.psrname
# logmass = np.float64(args.logmass)
# freq_mult = args.freqmult


psr = ds.Pulsar.read_feather(f'/DATA/CARINA_4/mit227/subband_PA_data/{psr_name}.PA.head_derot.median.median.mean.feather', json_metadata=False)
psr.set_position_from_ATNF()

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S") # Format as YYYYMMDD-HHMMSS (e.g., 20261027-114200)

fit_results_path = os.environ.get("FITRES")
run_write_path = f'{fit_results_path}/ppa_run_{timestamp}'
os.mkdir(run_write_path)

with open(f'{run_write_path}/axion_lims.csv', 'w', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(['log10(Axion mass/eV)', 
                     'log10(95% UB on axion PA amplitude/rad)', 
                     'log10(95% UB on axion-photon coupling/GeV^-1)'
                     ])

min_mass = ds.const.hbar_in_eVs * 2*np.pi / (psr.toas[-1] - psr.toas[0])
# log10_axion_mass_list = [np.log10((i+1)*min_mass) for i in range(100)]
log10_axion_mass_list = np.linspace(-23,-20,31)

first_mass = True
for log10_axion_mass in log10_axion_mass_list:
    axion_mass = 10**log10_axion_mass
    
    m_aldm = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, tnequad=True, log_efac=True),
                            ds.makegp_ecorr(psr),
                            # ds.makegp_erm(psr),
                            ds.makegp_fourier(psr, ds.powerlaw, components=180, fourierbasis=ds.fourierbasis_dm, name='rmgp'),
                            ds.makegp_aldm_psr(psr, log10_axion_mass)
                            ])
    logL = m_aldm.logL

    # static layout shared by every Params built inside the model
    layout, size = make_layout(logL.params)

    # flat uniform bounds, one entry per (scalar) parameter
    bounds = [prior.getprior_uniform(p) for p in logL.params]
    lo = jnp.array([b[0] for b in bounds])
    hi = jnp.array([b[1] for b in bounds])
    if first_mass:
        with open(f'{run_write_path}/priors.csv', 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            for i,param in enumerate(logL.params):
                writer.writerow([param, lo[i], hi[i]])

    def numpyro_model():
        raw = numpyro.sample('raw', dist.Uniform(lo, hi))   # one (P,) array site
        numpyro.factor('logl', logL(Params(raw, layout)))

    jax.debug.print(f'Starting run: m_a (eV) = {axion_mass}')
    mcmc = infer.MCMC(infer.NUTS(numpyro_model),# num_chains=4,
                        num_warmup=100, num_samples=600)
    mcmc.run(jax.random.PRNGKey(42))
    

    samples = mcmc.get_samples()
    raw_samples = samples['raw']

    np.save(f"{run_write_path}/mcmc_samples_{str(log10_axion_mass)}.npy", raw_samples)

    p_fit = {}
    for i, name in enumerate(logL.params):
        col = raw_samples[:, i]
        print(f'{name:35s} mean={col.mean():+.3f}  std={col.std():.3f}  bounds={bounds[i]}')
        p_fit[name] = col.mean()
    print(f'logL(mean params) = {logL(p_fit):e}\n')

    figure = corner.corner(
        np.array(raw_samples),
        labels=logL.params
        )
    plt.savefig(f"{run_write_path}/mcmc_samples_{str(axion_mass)}.pdf")

    log10_Sa_samples = raw_samples[:, 1]
    log10_Sa_95 = np.percentile(log10_Sa_samples, 95)
    print(f'95th percentile of log10(S_a/rad):   {log10_Sa_95}')
    Sa_95 = 10**np.percentile(log10_Sa_samples, 95)
    print(f'95th percentile of S_a (rad):        {Sa_95}')

    Sa_95_deg = np.rad2deg(Sa_95)
    log10_Sa_95_deg = np.log10(Sa_95_deg)
    print(f'95th percentile of log10(S_a/deg):   {log10_Sa_95_deg}')
    print(f'95th percentile of S_a (deg):        {Sa_95_deg}')

    coupling_95 = ds.Sa_to_coupling(Sa_95, axion_mass, DM_density=0.4)
    log10_coupling_95 = np.log10(coupling_95)
    print(f'Corresponding log coupling (GeV^-1): {log10_coupling_95}')
    print(f'Corresponding coupling (GeV^-1):     {coupling_95}\n')

    with open(f'{run_write_path}/axion_lims.csv', 'a', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([axion_mass, Sa_95, coupling_95])

    first_mass = False
