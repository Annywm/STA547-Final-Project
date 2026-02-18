data {
  int<lower=1> N;
  array[N] real t_obs;
  array[N] real y_obs;
  real dose;
}

parameters {
  real<lower=0> ka;
  real<lower=0> CL;
  real<lower=0> V_cent;
  real<lower=0> sigma;
  real<lower=0> Q;
  real<lower=0> V_peri;
}

transformed parameters {
  vector[N] c;
  matrix[3, 3] K;   //  (1=Gut, 2=Cent, 3=Peri)
  vector[3] y0;     // initial state
  
  K = rep_matrix(0, 3, 3);
  
  K[1, 1] = -ka;
  
  K[2, 1] = ka;
  K[2, 2] = -(CL/V_cent + Q/V_cent);
  K[2, 3] = Q/V_peri;
  
  K[3, 2] = Q/V_cent;
  K[3, 3] = -Q/V_peri;

  // initial state
  y0[1] = dose;
  y0[2] = 0;
  y0[3] = 0;

  // calculate concentration at each time point
  for (i in 1:N) {
    // calculate matrix exponential e^(Kt)
    vector[3] y_t = matrix_exp(K * t_obs[i]) * y0;
    
    // we can only observe the central compartment (second compartment)
    c[i] = y_t[2] / V_cent;
  }
}

model {
  CL ~ lognormal(log(10), 0.25);
  V_cent ~ lognormal(log(35), 0.25);
  ka ~ lognormal(log(2.5), 1);
  sigma ~ normal(0, 1);
  y_obs ~ lognormal(log(c), sigma);
  Q ~ lognormal(log(15), 0.5);
  V_peri ~ lognormal(log(105), 0.5);
}

generated quantities {
  array[N] real y_rep;
  vector[N] log_lik;
  for (i in 1:N) {
    y_rep[i] = lognormal_rng(log(c[i]), sigma);
    log_lik[i] = lognormal_lpdf(y_obs[i] | log(c[i]), sigma);
  }
}