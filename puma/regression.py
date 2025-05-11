import numpy as np



# Function to get mean and width of a peak in a histogrammed distribution
def get_mean_and_width(data, method='quantile_relative', weights=None):
    #weights=None # until numpy 2.0.0 works properly
    if method not in ['mean_std','quantile', 'quantile_relative', 'custom_quantile_weights']:
        raise ValueError("'method' argument must be one of 'mean_std', 'quantile','quantile_relative' or 'custom_quantile_weights'")

    # Compute mean and standard deviation
    elif method == "mean_std":
        return np.mean(data), np.std(data)

    # Compute half the central 68.2% quantile
    elif method == 'quantile':
        if len(data) == 0:
            return 0, 0
        
        quantile_method = "linear" if weights is None else "inverted_cdf"

        # median is the 50% quantile
        median = np.quantile(data, 0.5, method=quantile_method) #, weights=weights)

        # Find the -1 sigma (15.9th percentile) and +1 sigma (84.1st percentile) values,
        # which corresponds to half the central 68.2% quantile
        plus_one_sigma = np.quantile(data, 0.841, method=quantile_method) #, weights=weights)
        minus_one_sigma = np.quantile(data, 0.159, method=quantile_method) #, weights=weights)

        # Calculate the difference between +1 and -1 sigma percentiles
        central_half_quantile = (plus_one_sigma - minus_one_sigma) / 2

        return median, central_half_quantile
    
    elif method == "quantile_relative":
        if len(data) == 0:
            return 0, 0
        
        quantile_method = "linear" if weights is None else "inverted_cdf"

        # median is the 50% quantile
        median = np.quantile(data, 0.5, method=quantile_method) #, weights=weights)

        # Find the -1 sigma (15.9th percentile) and +1 sigma (84.1st percentile) values,
        # which corresponds to half the central 68.2% quantile
        plus_one_sigma = np.quantile(data, 0.841, method=quantile_method) #, weights=weights)
        minus_one_sigma = np.quantile(data, 0.159, method=quantile_method) #, weights=weights)

        # Calculate the difference between +1 and -1 sigma percentiles
        central_half_quantile = (plus_one_sigma - minus_one_sigma) / (2 * median)

        return median, central_half_quantile
    
    elif method == "custom_quantile_weights":
        values, bin_edges = np.histogram(data, weights=weights, density=True, bins=1000)
        # make histogram between min and max, 100 bins (with weights)

        bin_width = bin_edges[1]-bin_edges[0]
        minus_one_sigma = -1
        plus_one_sigma = -1
        median = -1
        running_total = 0
        for i in range(len(bin_edges)-1):
            running_total += values[i]
            if (minus_one_sigma < 0)  and (running_total > 0.159*np.sum(values)):
                minus_one_sigma = bin_edges[i]+0.5*bin_width
            if (median < 0) and (running_total > 0.5*np.sum(values)):
                median = bin_edges[i]+0.5*bin_width
            if (plus_one_sigma < 0)  and (running_total > 0.841*np.sum(values)):
                plus_one_sigma = bin_edges[i]+0.5*bin_width

        # Calculate the difference between +1 and -1 sigma percentiles
        central_half_quantile = (plus_one_sigma - minus_one_sigma) / (2 * median)

        return median, central_half_quantile


def bootstrap_uncertainties(data, unc_func, n_subsamples=10, weights=None, random_seed=42):

    def _unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    np.random.seed(random_seed)

    if weights is not None:
        shuffled_data, shuffled_weights = _unison_shuffled_copies(data, weights)
    else:
        shuffled_data = np.copy(data)
        np.random.shuffle(shuffled_data)
        shuffled_weights = None
    data_subsets = [shuffled_data[i::n_subsamples] for i in range(n_subsamples)]
    weights_subsets = [shuffled_weights[i::n_subsamples] for i in range(n_subsamples)] if weights is not None else None
    means, sigmas = [], []
    #for subset in data_subsets:
    for i in range(len(data_subsets)):
        subset = data_subsets[i] 
        subset_weights = weights_subsets[i] if weights is not None else None
        if subset_weights is not None:
            mean, sigma = unc_func(subset, weights=subset_weights)
        else:
            mean, sigma = unc_func(subset)
        means.append(mean)
        sigmas.append(sigma)
    mean_unc = np.std(means)/2   # diving by 2 for plotting error bars
    sigma_unc = np.std(sigmas)/2 # diving by 2 for plotting error bars
    return mean_unc, sigma_unc