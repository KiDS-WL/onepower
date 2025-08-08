import collections
import matplotlib.pyplot as pl
import numpy as np
import scipy.interpolate
from astropy.units import Quantity
from cosmosis.datablock import names, option_section
from glob import glob


def load_covariance_2d(covfile, nobsbins, nrbins, exclude=None):
    """Load covariance from a 2d matrix-like covariance file

    Parameters
    ----------
    covfile : str
        filename of covariance file
    exclude : list-like, optional
        bins to be excluded from the covariance. Only supports one set
        of bins, which is excluded from the entire dataset

    Returns
    -------
    cov : array, shape (Nobsbins,Nobsbins,Nrbins,Nrbins)
        four-dimensional covariance matrix
    icov : array, shape (Nobsbins,Nobsbins,Nrbins,Nrbins)
        four-dimensional inverse covariance matrix
    likenorm : float
        constant additive term for the likelihood
    esd_err : array, shape (Nobsbins,Nrbins)
        square root of the diagonal elements of the covariance matrix
    cov2d : array, shape (Nobsbins*Nrbins,Nobsbins*Nrbins)
        re-shaped covariance matrix, for plotting or other uses
    """
    cov2d = np.loadtxt(covfile)
    cov2d, nrbins = covariance_excluder(cov2d, nobsbins, nrbins, exclude)
    cov, icov, likenorm, signal_err, cor = covariance_aux_numbers(cov2d, nobsbins, nrbins)

    return cov, icov, likenorm, signal_err, cov2d, cor


def covariance_aux_numbers(cov2d, nobsbins, nrbins):
    try:
        icov_in = np.linalg.inv(cov2d)
    except:
        warnings.warn(
            'Standard matrix inversion failed, using the Moore-Penrose' \
                ' pseudo-inverse',
            RuntimeWarning)
        icov_in = np.linalg.pinv(cov2d)
    cor_in = cov2d/np.sqrt(np.outer(np.diag(cov2d), np.diag(cov2d.T)))
    cov = []
    icov = []
    cor = []
    signal_err = []
    for i in range(nobsbins):
        cov.append([])
        icov.append([])
        cor.append([])
        for j in range(nobsbins):
            indices_i = nrbins[:i].sum()
            indices_j = nrbins[:j].sum()
            s_ = np.s_[indices_i:indices_i+nrbins[i], indices_j:indices_j+nrbins[j]]
            cov[-1].append(cov2d[s_])
            icov[-1].append(icov_in[s_])
            cor[-1].append(cor_in[s_])
            if i==j:
                signal_err.append(np.diag(cov2d[s_])**0.5) # This needs to be in shape (nbins,)
    # product of the determinants
    prod_detC = np.linalg.det(cov2d)
    try:
        posdet = np.linalg.cholesky(cov2d)
    except np.linalg.LinAlgError:
        print('WARNING: Covariance matrix is not positive definite!')
    # likelihood normalization
    likenorm = -(nobsbins**2*np.log(2*np.pi) + np.log(prod_detC)) / 2
    return cov, icov, likenorm, signal_err, cor


def covariance_excluder(cov2d, nobsbins, nrbins, exclude):
    if exclude is None:
        return cov2d, nrbins
    if not hasattr(exclude, '__iter__'):
        exclude = [exclude]
    nexcl = len(exclude)
    idx = np.array([[nrbins[:a].sum()+b for b in exclude] for a in range(nobsbins)])
    cov2d = np.delete(cov2d, idx, axis=0)
    cov2d = np.delete(cov2d, idx, axis=1)
    for i,n in enumerate(nrbins):
        if len(exclude[exclude >= n]) != 0:
            nrbins[i] = n - len(exclude[exclude < n])
        else:
            nrbins[i] = n - nexcl
    return cov2d, nrbins


def load_datapoints_2d(datafiles, datacols, exclude=None):

    datafiles = sorted(glob(datafiles))
    nobsbins = len(datafiles)
    x = np.empty(nobsbins, dtype=object)
    y = np.empty(nobsbins, dtype=object)
    for i,file in enumerate(datafiles):
        x[i] = np.loadtxt(file, usecols=datacols[0])
        y[i] = np.loadtxt(file, usecols=datacols[1])

    if len(datacols) == 3:
        oneplusk = np.empty(nobsbins, dtype=object)
        for i,file in enumerat(datafiles):
            oneplusk[i] = np.loadtxt(file, usecols=[datacols[2]])
        y /= oneplusk
    if exclude is not None:
        x = np.array([np.array([Ri[j] for j in range(len(Ri))
                      if j not in exclude]) for Ri in x], dtype=object)
        y = np.array([np.array([esdi[j] for j in range(len(esdi))
                        if j not in exclude]) for esdi in y], dtype=object)
    return x, y, nobsbins


def load_data(data_dict):

    # first load without excluding anything to define Nrbins, Nobsbins
    x, y, nobsbins = load_datapoints_2d(data_dict['data'], data_dict['columns'])
    nrbins = np.array([sh.size for sh in y])
    cov = load_covariance_2d(
        data_dict['covariance'],nobsbins, nrbins, data_dict['exclude'])
    # Now exclude data points
    x, y, nobsbins = load_datapoints_2d(
        data_dict['data'], data_dict['columns'], data_dict['exclude'])
    nrbins = np.array([sh.size for sh in y])

    return x, y, cov, nobsbins, nrbins


def get_theory_point(x, y, mode='interpolate', interpolated_x=None, bin_edges=None, weighting=None):
    intp = scipy.interpolate.interp1d(np.log(x), y, kind='linear')
    if mode == 'interpolate':
        result = intp(np.log(interpolated_x))
    elif mode == 'integrate':
        result = np.zeros(bin_edges.size-1)
        for i in range(bin_edges.size-1):
            mask = (bin_edges[i] <= x) & (x < bin_edges[i+1])
            if isinstance(weighting, np.ndarray):
                w = weighting[mask]
                norm = np.trapz(w, x[mask])
            else:
                w = 1.0
                norm = 1.0
            result[i] = np.trapz(w*y[mask], x[mask])/norm
    else:
        raise ValueError(f'Mode {mode} not supported.')
    return result




def setup(options):

    # TO-DO: - Implement exclude option
    #        - Implement scaling of observables to the cosmology of the data
    #        -

    input_sections = options.get_string(option_section, 'input_section_names').split()

    data_dict = {}
    data_dict['data'] = options[option_section, 'data']
    data_dict['covariance'] = options[option_section, 'covariance']
    data_dict['columns'] = np.asarray([options[option_section, 'columns']]).flatten()
    #data_dict['exclude'] = np.asarray([options[option_section, 'exclude']]).flatten() # Maybe do with classic scale-cuts as well?
    data_dict['exclude'] = None#options.get_bool(option_section, 'exclude', default=None)# Maybe do with classic scale-cuts as well?
    unit = options[option_section, 'unit']

    x, data_vectors, cov, nobsbins, nrbins = load_data(data_dict)
    cov, inv_cov, likenorm, err, cov2d, cor = cov

    binning_mode = options.get_string(option_section, 'binning_mode', default='interpolate')
    if binning_mode == 'integrate':
        edges = np.array([np.empty(xi.size+1) for xi in x], dtype=object)
        for i,_ in enumerate(edges):
            xi = x[i].astype(float)
            edges[i][1:-1] = (np.log10(xi[1:]) + np.log10(xi[:-1])) * 0.5
            # Compute the first and last by making them symmetric
            edges[i][0] = 2.0 * np.log10(xi[0]) - edges[i][1]
            edges[i][-1] = 2.0 * np.log10(xi[-1]) - edges[i][-2]
        edges = 10.0**edges
    else:
        edges = x.copy()

    like_name = options.get_string(option_section, 'like_name')
    keep_theory_vector = options.get_bool(option_section, 'keep_theory_vector', False)
    return input_sections, cov, inv_cov, x, data_vectors, nobsbins, nrbins, binning_mode, edges, like_name, keep_theory_vector, unit


def execute(block, config):
    input_sections, cov, inv_cov, x, data_vectors, nobsbins, nrbins, binning_mode, edges, like_name, keep_theory_vector, unit = config

    nbins = 0
    for input in input_sections:
        nbins += block[input, 'nbin']
    if nbins != nobsbins:
        raise ValueError('Number of bins in data vector is not the same as the total number of theory bins!')

    count = 0
    theory_vectors = np.empty(nobsbins, dtype=object)
    for i,input in enumerate(input_sections):
        nbins = block[input, 'nbin']
        for j in range(nbins):
            if input in ['wp', 'ds']:
                data_xi = x[count].astype(float)
                data_xi = Quantity(data_xi, unit=unit)
                data_x = data_xi.to('Mpc').value
                theory_x_in = block[input, 'rp']
            else:
                data_x = x[count].astype(float)
                theory_x_in = block[input, f'obs_{j+1}'] * block['cosmological_parameters', 'h0']
            theory_y_in = block[input, f'bin_{j+1}']

            theory_vectors[count] = get_theory_point(theory_x_in, theory_y_in,
                                            mode=binning_mode,
                                            interpolated_x=data_x,
                                            bin_edges=edges[count])
            #pl.plot(data_x, data_vectors[count])
            #pl.plot(data_x, theory_vectors[count])
            #pl.plot(data_x, data_vectors[count]/theory_vectors[count])
            count += 1

        #pl.xscale('log')
        #pl.yscale('log')
        #pl.show()
        #pl.clf

    residuals = data_vectors - theory_vectors
    chi2 = np.array([np.dot(residuals[m], np.dot(inv_cov[m][n], residuals[n]))
                    for m in range(nobsbins) for n in range(nobsbins)]).sum()
    like = -0.5*chi2

    block[names.data_vector, like_name+'_CHI2'] = chi2
    block[names.likelihoods, like_name+'_LIKE'] = like

    if keep_theory_vector:
        block[names.data_vector, 'theory'] = np.hstack(theory_vectors.flatten())
        block[names.data_vector, 'data'] = np.hstack(data_vectors.flatten())
        #block[names.data_vector, like_name + "_inverse_covariance"] = inv_cov.flatten()

    return 0

def cleanup(config):
    pass
