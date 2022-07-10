import numpy as np
import pandas as pd
from sklearn import mixture
from scipy.stats import norm
import matplotlib.pyplot as plt


def read_stat(data_dir):
    df = pd.ExcelFile(data_dir+'.xlsx')
    df_dates_list = []
    # read stats date by date
    for sheet_name in df.sheet_names:
        df_data = df.parse(sheet_name, skiprows=0)
        df_dates_list.append(df_data)
    df_dates = pd.concat(df_dates_list, axis=0)
    return df_dates


def prepare_data(data_dir):
    df_1 = read_stat(data_dir + '/period_1')
    duration_1 = df_1.loc[:, 'duration'].values
    df_2 = read_stat(data_dir + '/period_2')
    duration_2 = df_2.loc[:, 'duration'].values
    df_3 = read_stat(data_dir + '/period_3')
    duration_3 = df_3.loc[:, 'duration'].values

    duration_123 = list(duration_1) + list(duration_2) + list(duration_3)
    duration_123 = np.array(duration_123).reshape(-1, 1)
    return duration_123


def gauss(x, mu, sigma, a):
    y = norm.pdf(x, mu, sigma)*a
    return y


def bimodal(x, mu1, sigma1, mu2, sigma2):
    return gauss(x, mu1, sigma1) + gauss(x, mu2, sigma2)


def gmm_modeling(dataset):

    # fitting
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    gmm = gmm.fit(dataset)

    # prediction
    y = gmm.predict(dataset)

    # parameters
    means = gmm.means_.reshape(2, )
    var = gmm.covariances_.reshape(2, )
    stds = np.sqrt(var)
    weights = gmm.weights_

    idx_urine = np.argmin(means)
    idx_feces = np.argmax(means)
    num_urine = np.sum([y == idx_urine])
    num_feces = np.sum([y == idx_feces])

    threshold = (np.max(dataset[y == idx_urine])
                 + np.min(dataset[y == idx_feces])) // 2

    print("Numer of Urination Events: ", num_urine)
    print("Numer of Defecation Events: ", num_feces)

    # plotting
    gmm_x = np.linspace(0, 450)
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
    # Plot histograms and gaussian curves
    fig, ax = plt.subplots()
    ax.hist(dataset, bins=range(15, 450 + 5, 5), density=True)
    ax.plot(gmm_x, gmm_y, color="crimson", lw=1, label="GMM")

    label_1 = 'x~p(x|{:.0f}, {:.0f}\u00b2)*{:.1f}'.format(
        means[idx_urine], stds[idx_urine], weights[idx_urine])
    plt.plot(gmm_x, gauss(gmm_x, means[idx_urine], stds[idx_urine],
             weights[idx_urine]), color='red', lw=1, ls="--", label=label_1)

    label_2 = 'x~p(x|{:.0f}, {:.0f}\u00b2)*{:.1f}'.format(
        means[idx_feces], stds[idx_feces], weights[idx_feces])
    plt.plot(gmm_x, gauss(gmm_x, means[idx_feces], stds[idx_feces],
             weights[idx_feces]), color='red', lw=1, ls=":", label=label_2)

    plt.axvline(x=threshold, ls='-', linewidth=1,
                color='red', label='t = %d' % threshold)
    ax.set_xlabel("Duration (s)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('gmm.png')


def main():
    data_dir = './data'
    dataset = prepare_data(data_dir)
    gmm_modeling(dataset)


if __name__ == '__main__':
    main()
