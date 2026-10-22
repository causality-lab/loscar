# R packages for the SID metric (cdt.metrics.SID / SID_CPDAG), used by all
# multivariate experiments. graph and RBGL (Bioconductor) come from conda,
# see environment.yaml. Run with: Rscript setupR.R
repos <- "https://cloud.r-project.org"
install.packages(c("ggm", "pcalg", "SID"), repos = repos, dependencies = TRUE)
