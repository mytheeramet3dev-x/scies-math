# `inference_ext` Module Documentation

Extended statistical inference: non-parametric tests, multiple comparison
corrections, bootstrap, permutation tests, additional correlations.

| Test | Function |
|---|---|
| Mann-Whitney U | `mann_whitney_u` |
| Wilcoxon signed-rank | `wilcoxon_signed_rank` |
| Kruskal-Wallis | `kruskal_wallis` |
| Kolmogorov-Smirnov 1-sample | `ks_test_1sample` |
| Kolmogorov-Smirnov 2-sample | `ks_test_2sample` |
| Fisher's exact | `fisher_exact_2x2` |
| Levene's test | `levene_test` |
| Spearman correlation | `spearman_correlation` |
| Kendall's τ | `kendall_tau` |
| Bootstrap CI | `bootstrap_ci` |
| Permutation test | `permutation_test` |
| Bonferroni / Holm / BH | `p_adjust` |