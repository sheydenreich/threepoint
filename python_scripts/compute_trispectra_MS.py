from utility import trispectrum_extractor
from file_loader import get_kappa_millennium
import numpy as np
from tqdm import trange


k_array = np.geomspace(150,40000,14)
for i in trange(64):
    kappa_map = get_kappa_millennium(i)
    te = trispectrum_extractor(kappa_map)
    tri_equilateral = te.extract_trispectra(k_array,only_diagonal=True)
    result = np.array([k_array,tri_equilateral]).T
    np.savetxt("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/trispectra/trispectrum_equilateral_los_{}.dat".format(i),
                result)