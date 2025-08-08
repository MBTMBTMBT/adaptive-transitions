import numpy as np
from typing import Union
from scipy.stats import norm

from .mdp_network import MDPNetwork
from .mdp_tables import ValueTable


def gaussian_bayesian_update(prior_mu: Union[float, np.ndarray],
                             prior_sigma: Union[float, np.ndarray],
                             observation: Union[float, np.ndarray],
                             observation_sigma: Union[float, np.ndarray],
                             min_sigma: float = 1e-8) -> tuple[
    Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Bayesian update for Gaussian distributions with numerical stability.

    Args:
        prior_mu: Prior mean (float or numpy array)
        prior_sigma: Prior standard deviation (float or numpy array)
        observation: New observation value (float or numpy array)
        observation_sigma: Observation noise standard deviation (float or numpy array)
        min_sigma: Minimum allowed standard deviation to prevent numerical issues

    Returns:
        Tuple of (posterior_mu, posterior_sigma)
        - posterior_mu: Updated mean (same type as input)
        - posterior_sigma: Updated standard deviation (same type as input)

    Note:
        Uses conjugate prior formula for Gaussian-Gaussian update:
        posterior_precision = prior_precision + observation_precision
        posterior_mu = (prior_precision * prior_mu + observation_precision * observation) / posterior_precision
        Handles numerical edge cases with minimum sigma threshold.
    """
    # Ensure minimum sigma to prevent division by zero
    prior_sigma = np.maximum(prior_sigma, min_sigma)
    observation_sigma = np.maximum(observation_sigma, min_sigma)

    # Convert to precision (inverse variance)
    prior_precision = 1.0 / (prior_sigma ** 2)
    obs_precision = 1.0 / (observation_sigma ** 2)

    # Compute posterior precision and variance
    posterior_precision = prior_precision + obs_precision
    posterior_variance = 1.0 / posterior_precision
    posterior_sigma = np.sqrt(posterior_variance)

    # Compute posterior mean
    posterior_mu = (prior_precision * prior_mu + obs_precision * observation) / posterior_precision

    # Ensure posterior sigma is not too small
    posterior_sigma = np.maximum(posterior_sigma, min_sigma)

    return posterior_mu, posterior_sigma


def gaussian_kl_divergence(mu1: Union[float, np.ndarray],
                           sigma1: Union[float, np.ndarray],
                           mu2: Union[float, np.ndarray],
                           sigma2: Union[float, np.ndarray],
                           min_sigma: float = 1e-8,
                           max_kl: float = 1e10) -> Union[float, np.ndarray]:
    """
    Compute KL divergence between two Gaussian distributions KL(P||Q) with numerical stability.

    Args:
        mu1: Mean of distribution P (float or numpy array)
        sigma1: Standard deviation of distribution P (float or numpy array)
        mu2: Mean of distribution Q (float or numpy array)
        sigma2: Standard deviation of distribution Q (float or numpy array)
        min_sigma: Minimum allowed standard deviation to prevent numerical issues
        max_kl: Maximum allowed KL divergence to prevent infinite values

    Returns:
        KL divergence KL(P||Q) (same type as input)

    Note:
        Formula: KL(P||Q) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        Returns element-wise KL divergence for array inputs.
        Handles numerical edge cases with minimum sigma and maximum KL thresholds.
    """
    # Ensure minimum sigma to prevent division by zero and log(0)
    sigma1 = np.maximum(sigma1, min_sigma)
    sigma2 = np.maximum(sigma2, min_sigma)

    # Compute variance terms
    var1 = sigma1 ** 2
    var2 = sigma2 ** 2

    # Compute mean difference
    mu_diff = mu1 - mu2

    # KL divergence formula for Gaussians
    # Handle potential numerical issues with log and division
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log(sigma2 / sigma1)
        variance_term = (var1 + mu_diff ** 2) / (2 * var2)
        kl_div = log_term + variance_term - 0.5

    # Handle edge cases and clip extreme values
    if np.isscalar(kl_div):
        if np.isnan(kl_div) or np.isinf(kl_div):
            # If distributions are identical, KL should be 0
            if np.isclose(mu1, mu2) and np.isclose(sigma1, sigma2):
                kl_div = 0.0
            else:
                kl_div = max_kl
        else:
            kl_div = np.clip(kl_div, 0.0, max_kl)
    else:
        # Array case
        # Replace NaN and Inf with appropriate values
        identical_mask = np.isclose(mu1, mu2) & np.isclose(sigma1, sigma2)
        kl_div = np.where(identical_mask, 0.0, kl_div)

        # Replace remaining NaN/Inf with max_kl
        kl_div = np.where(np.isnan(kl_div) | np.isinf(kl_div), max_kl, kl_div)

        # Clip to reasonable range
        kl_div = np.clip(kl_div, 0.0, max_kl)

    return kl_div


def compute_information_surprise(mdp: MDPNetwork,
                                 occupancy: ValueTable,
                                 prior_mu: float,
                                 prior_sigma: float,
                                 observation_sigma: float = 1e-8,
                                 delta: float = 1e-4) -> dict:
    """
    Compute information surprise for terminal states using both KL divergence and interval-based -logP surprise.

    Args:
        mdp: MDP network
        occupancy: State occupancy measure (ValueTable)
        prior_mu: Prior mean of reward distribution
        prior_sigma: Prior standard deviation of reward distribution
        observation_sigma: Standard deviation for reward observations
        delta: Precision width used to estimate probability interval for -logP

    Returns:
        Dictionary containing surprise analysis results (both KL and -logP)
    """
    terminal_states = list(mdp.terminal_states)
    if not terminal_states:
        return {'error': 'No terminal states found'}

    print(f"Computing information surprise for {len(terminal_states)} terminal states...")

    terminal_info = []
    total_weighted_kl = 0.0
    total_weighted_nll = 0.0

    for state in terminal_states:
        state_occupancy = occupancy.get_value(state)
        reward = mdp.get_state_reward(state)

        posterior_mu, posterior_sigma = gaussian_bayesian_update(
            prior_mu, prior_sigma, reward, observation_sigma
        )

        kl_divergence = gaussian_kl_divergence(
            posterior_mu, posterior_sigma, prior_mu, prior_sigma
        )

        # Compute interval probability under prior using CDF
        lower = reward - delta / 2
        upper = reward + delta / 2
        prob = norm.cdf(upper, loc=prior_mu, scale=prior_sigma) - norm.cdf(lower, loc=prior_mu, scale=prior_sigma)

        # Avoid log(0)
        nll = -np.log(max(prob, 1e-12))

        weighted_kl = kl_divergence * state_occupancy
        weighted_nll = nll * state_occupancy

        total_weighted_kl += weighted_kl
        total_weighted_nll += weighted_nll

        terminal_info.append({
            'state': state,
            'reward': reward,
            'occupancy': state_occupancy,
            'posterior_mu': posterior_mu,
            'posterior_sigma': posterior_sigma,
            'kl_divergence': kl_divergence,
            'weighted_kl': weighted_kl,
            'negative_log_likelihood': nll,
            'weighted_nll': weighted_nll
        })

        print(f"  State {state}: reward={reward:.4f}, occupancy={state_occupancy:.6f}, "
              f"KL={kl_divergence:.6f}, weighted_kl={weighted_kl:.6f}, "
              f"-logP={nll:.6f}, weighted_nll={weighted_nll:.6f}")

    terminal_info.sort(key=lambda x: x['weighted_nll'], reverse=True)

    results = {
        'prior_mu': prior_mu,
        'prior_sigma': prior_sigma,
        'observation_sigma': observation_sigma,
        'delta': delta,
        'total_information_surprise_kl': total_weighted_kl,
        'total_information_surprise_nll': total_weighted_nll,
        'num_terminal_states': len(terminal_states),
        'terminal_state_analysis': terminal_info,
        'max_surprise_state': terminal_info[0]['state'] if terminal_info else None,
        'max_surprise_value_nll': terminal_info[0]['weighted_nll'] if terminal_info else 0.0,
        'max_surprise_value_kl': terminal_info[0]['weighted_kl'] if terminal_info else 0.0
    }

    print(f"Total Information Surprise (KL): {total_weighted_kl:.6f}")
    print(f"Total Information Surprise (-logP): {total_weighted_nll:.6f}")
    print(f"Most surprising state (by -logP): {results['max_surprise_state']} "
          f"(surprise={results['max_surprise_value_nll']:.6f})")

    return results
