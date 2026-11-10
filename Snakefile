"""Snakemake workflow for cfchip-sim analytical validation."""

# Configuration
REPS = config.get("reps", 20)
TUMOR_LINES = ["MCF-7", "A549", "HepG2", "K562"]

# Default target
rule all:
    input:
        "data/processed/chip_fc.tsv",
        "results/baseline.tsv",
        "results/sim_results.tsv",
        "results/validation_summary.tsv",
        "results/peaks_vs_degs.tsv",
        "results/signature_vs_tf.tsv",
        "figures/fig1_baseline.png",
        "figures/fig2_tf_estimate.png",
        "figures/fig3_call_accuracy.png",
        "figures/fig4_roc.png",
        "figures/fig5_peaks_vs_degs.png",
        "figures/fig6_signature_vs_tf.png"


rule fetch:
    """Resolve ENCODE accessions -> data/raw/manifest.tsv (review it!)"""
    output:
        "data/raw/manifest.tsv"
    shell:
        "python scripts/01_fetch_encode.py"


rule download:
    """Download files in the manifest + GENCODE v29 GTF"""
    input:
        "data/raw/manifest.tsv"
    output:
        "data/ref/gencode.v29.primary_assembly.annotation.gtf.gz"
    shell:
        "python scripts/01_fetch_encode.py --download"


rule matrices:
    """Build H3K4me3 and RNA-seq matrices from downloaded data"""
    input:
        manifest = "data/raw/manifest.tsv",
        gtf = "data/ref/gencode.v29.primary_assembly.annotation.gtf.gz"
    output:
        chip = "data/processed/chip_fc.tsv",
        rna = "data/processed/rna_tpm.tsv",
        promoters = "data/processed/promoters.tsv"
    shell:
        "python scripts/02_build_matrices.py"


rule baseline:
    """Figure 1: promoter H3K4me3 vs RNA in pure lines"""
    input:
        chip = "data/processed/chip_fc.tsv",
        rna = "data/processed/rna_tpm.tsv"
    output:
        results = "results/baseline.tsv",
        figure = "figures/fig1_baseline.png"
    shell:
        "python scripts/03_baseline.py"


rule simulate:
    """Simulate plasma dilution and tumor-fraction estimation"""
    input:
        chip = "data/processed/chip_fc.tsv",
        rna = "data/processed/rna_tpm.tsv"
    output:
        sim = "results/sim_results.tsv",
        roc = "results/roc_curves.tsv"
    params:
        reps = REPS
    shell:
        "python scripts/04_simulate.py --reps {params.reps}"


rule report:
    """Figures 2-4 and validation summary table"""
    input:
        sim = "results/sim_results.tsv",
        roc = "results/roc_curves.tsv",
        baseline = "results/baseline.tsv"
    output:
        summary = "results/validation_summary.tsv",
        fig2 = "figures/fig2_tf_estimate.png",
        fig3 = "figures/fig3_call_accuracy.png",
        fig4 = "figures/fig4_roc.png"
    shell:
        "python scripts/05_report.py"


rule peaks:
    """Figure 5: gene-level recovery, precision@100 for tumor-specific peaks vs DEGs"""
    input:
        chip = "data/processed/chip_fc.tsv",
        rna = "data/processed/rna_tpm.tsv"
    output:
        results = "results/peaks_vs_degs.tsv",
        figure = "figures/fig5_peaks_vs_degs.png"
    shell:
        "python scripts/05b_peaks_vs_degs.py"


rule signatures:
    """Figure 6: pathway-level validation, top Hallmark signature vs tumor fraction"""
    input:
        chip = "data/processed/chip_fc.tsv",
        rna = "data/processed/rna_tpm.tsv"
    output:
        results = "results/signature_vs_tf.tsv",
        figure = "figures/fig6_signature_vs_tf.png"
    shell:
        "python scripts/06_signature_vs_tf.py"


rule test:
    """Run pytest unit tests"""
    shell:
        "python -m pytest -q tests"


rule demo:
    """Whole pipeline on synthetic data, no downloads (~10 s)"""
    output:
        demo_chip = "data/demo/processed/chip_fc.tsv",
        demo_rna = "data/demo/processed/rna_tpm.tsv",
        demo_fig1 = "figures/demo/fig1_baseline.png",
        demo_summary = "results/demo/validation_summary.tsv"
    shell:
        """
        python scripts/make_demo_data.py
        python scripts/03_baseline.py --processed data/demo/processed --figures figures/demo --results results/demo
        python scripts/04_simulate.py --processed data/demo/processed --results results/demo --reps 10
        python scripts/05_report.py --results results/demo --figures figures/demo
        """


rule clean:
    """Remove generated data, results, and figures"""
    shell:
        "rm -rf data/processed data/demo results figures"
