EPS        = 1e-8   # epsilon (a small value, mostly to avoid divisions by 0)
SEED       = 1234   # randomness seed
EPOCHS     = 10     # default number of training epochs
LR         = 0.001  # learning rate
BATCH_SZ   = 16     # batch size
ADV_EPS    = 0.12   # epsilon, the L_inf-norm bound of attacks
PGD_ALPHA  = 0.01   # alpha, the step size of PGD
PGD_ITERS  = 25     # # iters of PGD
