# TransferMod v1.2.2

## Added

- uniform runtime deprecation warnings for every v1.x certification facade;
- trained random-feature neural diffusion-operator validation;
- 2D/3D uniform-versus-adaptive PDE grid-scaling validation;
- Bayesian posterior Silent Risk validation;
- reproducible JSON outputs and `VALIDATION_REPORT.md`.

## Scope

The learned-operator validation uses a model trained within the repository, not
an external pretrained checkpoint. The PDE scaling validation uses analytic
localized fields and spectral truncation, not a production CFD solve. The
Bayesian validation uses a conjugate posterior, not MCMC. These are executable
first validations, with external and larger-scale replications retained in the
roadmap.

## 1.2.2 CI compatibility patch

Python 3.10 support is restored and the workflow uses Node.js 24-native GitHub Actions.
