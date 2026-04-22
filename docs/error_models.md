For fluorescence plate reader the error comes from:

- readout noise (= √ a)
- Poisson gain (= g)
- multiplicative noise coefficient (= √ c)

Var(y) = a + g × mean(y) + c × mean(y)²

Var(y) = σ_readout² + g × signal + × signal²

Procedural options:

1. variance-mean analysis i.e. plotting variance vs. mean from repeated measurements
1. residual analysis i.e. after fitting many curves plot |residual| or residual² vs predicted signal
   - std res r = (y - ŷ) / y_err
   - plot |r| vs. ŷ that should be flat ≈ 0.8

additions were calibrated to obtain robustness and consistency (see figure S1)
after each addition pH was measure in 3-4 protein samples to obtain a prior mean +/- SD
heuristic Bayesian model determined the exact pH values for each well after each addition,
i.e. the big heuristic modeling common starting pH + almost common additions.
the CTR should have the same K of course, but not sure this is a correct constraint; for instance leaving free independent K for CTRs would inform on the accuracy for all other samples.
