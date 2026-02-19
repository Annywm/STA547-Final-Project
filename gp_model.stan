functions {
  // squared exponential (exp-quad) covariance
  matrix cov_se(array[] real x, real alpha, real rho) {
    int N = size(x);
    matrix[N, N] K;
    for (i in 1:N) {
      K[i, i] = square(alpha);
      for (j in (i+1):N) {
        real sqdist = square((x[i] - x[j]) / rho);
        K[i, j] = square(alpha) * exp(-0.5 * sqdist);
        K[j, i] = K[i, j];
      }
    }
    return K;
  }
}

data {
  int<lower=1> N;
  array[N] real t_obs;
  array[N] real<lower=0> y_obs;
  real<lower=0> dose; // kept for interface consistency; not used
}

transformed data {
  vector[N] logy = log(to_vector(y_obs));
  real delta = 1e-9; // jitter
}

parameters {
  real a;                 // mean on log-scale
  real<lower=0> rho;      // length-scale
  real<lower=0> alpha;    // GP marginal sd
  real<lower=0> sigma;    // noise sd on log-scale
  vector[N] z;            // latent standard normal
}

transformed parameters {
  matrix[N, N] K = cov_se(t_obs, alpha, rho);
  matrix[N, N] L_K;
  vector[N] f;

  for (n in 1:N)
    K[n, n] = K[n, n] + delta;

  L_K = cholesky_decompose(K);
  f = L_K * z;            // GP latent function values
}

model {
  // priors (先用宽松但稳定的)
  z     ~ std_normal();
  a     ~ normal(0, 5);
  rho   ~ inv_gamma(5, 5);
  alpha ~ std_normal();   // half-normal since alpha>0
  sigma ~ std_normal();   // half-normal since sigma>0

  // likelihood on log-scale => y_obs is lognormal
  logy ~ normal(a + f, sigma);
}

generated quantities {
  vector[N] log_lik;
  array[N] real y_rep;

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(logy[n] | a + f[n], sigma);
    y_rep[n] = lognormal_rng(a + f[n], sigma);
  }
}