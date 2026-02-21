// Two-compartment + first-order absorption
// NONMEM event table + lognormal observation model

data {
  int<lower=1> nt;                 // total event rows (dose + obs)
  int<lower=1> nObs;               // number of observation rows
  array[nObs] int<lower=1> iObs;   // row indices of observations in 1:nt

  // NONMEM-style event table columns
  array[nt] int<lower=1> cmt;      // compartment index (1=gut, 2=central, 3=peri)
  array[nt] int evid;             // 1=dose/event, 0=observation
  array[nt] int addl;             // additional doses
  array[nt] int ss;               // steady-state flag
  array[nt] real amt;             // amount dosed (mg)
  array[nt] real time;            // time (h)
  array[nt] real rate;            // infusion rate (0 for bolus)
  array[nt] real ii;              // inter-dose interval (h)

  vector<lower=0>[nObs] cObs;      // observed concentration
}

transformed data {
  vector[nObs] logCObs = log(cObs);
  int nTheta = 5;
  int nCmt = 3;
}

parameters {
  // individual parameters (single subject)
  real<lower=0> CL;
  real<lower=0> Q;
  real<lower=0> V1;     // central volume
  real<lower=0> V2;     // peripheral volume
  real<lower=0> ka;

  // lognormal residual SD (on log scale)
  real<lower=0> sigma;
}

transformed parameters {
  array[nTheta] real theta;
  matrix[nCmt, nt] x;          // amounts in compartments over event rows
  row_vector[nt] cHat;         // predicted central concentration over event rows
  vector[nObs] cHatObs;        // predicted concentration at observation rows

  theta[1] = CL;
  theta[2] = Q;
  theta[3] = V1;
  theta[4] = V2;
  theta[5] = ka;

  // Torsten analytic solver for 2-cpt with first-order absorption
  x = pmx_solve_twocpt(time, amt, rate, ii, evid, cmt, addl, ss, theta);

  // central concentration = amount in central / V1
  cHat = fmax(x[2, :], 0) ./ V1;

  // pick out observation rows
  cHatObs = to_vector(cHat[iObs]);
}

model {
  // Priors (match your simulation settings)
  CL ~ lognormal(log(10), 0.25);
  Q  ~ lognormal(log(15), 0.5);
  V1 ~ lognormal(log(35), 0.25);
  V2 ~ lognormal(log(105), 0.5);
  ka ~ lognormal(log(2.5), 1);

  // You used abs(N(0,1)) in simulation; common fitting prior:
  sigma ~ cauchy(0, 1);

  // Likelihood: lognormal noise (same as cObs = cTrue * exp(N(0,sigma)))
  logCObs ~ normal(log(fmax(cHatObs, 1e-12)), sigma);
}

generated quantities {
  vector[nObs] log_lik;
  vector[nObs] y_rep; // 统一命名为 y_rep，用于 PPC

  for (i in 1:nObs) {
    // 提前计算好 log_chat，避免 normal_lpdf 和 normal_rng 重复计算
    real log_chat = log(fmax(cHatObs[i], 1e-12)); 
    log_lik[i] = normal_lpdf(logCObs[i] | log_chat, sigma);
    y_rep[i] = exp(normal_rng(log_chat, sigma));
  }
}

