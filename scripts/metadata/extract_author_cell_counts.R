#!/usr/bin/env Rscript

# Extract compact per-target cell counts from author RDS objects. Large source
# objects are inputs only and should be kept outside the repository.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(
    paste(
      "usage: extract_author_cell_counts.R MODE TARGET_FILE OUTPUT_DIR INPUT...",
      "MODE must be jiang or mcfaline"
    )
  )
}

mode <- args[[1]]
target_file <- args[[2]]
output_dir <- args[[3]]
input_files <- args[4:length(args)]
targets <- unique(toupper(trimws(readLines(target_file, warn = FALSE))))
targets <- targets[nzchar(targets) & !startsWith(targets, "#")]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_counts <- function(context, counts, prefix) {
  selected <- setNames(
    vapply(
      targets,
      function(target) {
        value <- counts[target]
        if (is.na(value)) 0L else as.integer(value)
      },
      integer(1)
    ),
    targets
  )
  selected <- selected[order(names(selected))]
  output <- data.frame(
    source_target_gene = names(selected),
    n_cells = as.integer(selected),
    stringsAsFactors = FALSE
  )
  path <- file.path(
    output_dir,
    paste0(prefix, "_", tolower(context), "_counts.csv")
  )
  write.csv(output, path, row.names = FALSE, quote = FALSE)
  message(context, ": wrote ", nrow(output), " target counts to ", path)
}

audit_rows <- list()
record_audit <- function(dataset, context, metadata, gene_column, controls) {
  assignments <- as.character(metadata[[gene_column]])
  controls_present <- assignments %in% controls | is.na(assignments)
  combinations <- !is.na(assignments) & grepl("[,;+]", assignments)
  audit_rows[[length(audit_rows) + 1]] <<- data.frame(
    dataset = dataset,
    context = context,
    n_cells_in_representation = nrow(metadata),
    assignment_column = gene_column,
    control_labels = paste(controls, collapse = ";"),
    n_control_or_missing = sum(controls_present),
    n_multiple_assignment_labels = sum(combinations),
    n_single_target_cells = sum(
      !controls_present & !combinations & toupper(assignments) %in% targets
    ),
    stringsAsFactors = FALSE
  )
}

if (mode == "mcfaline") {
  suppressPackageStartupMessages(library(SingleCellExperiment))
  if (length(input_files) != 1) {
    stop("mcfaline mode accepts exactly one preprocessed CDS-list RDS")
  }
  # GEO adds an outer gzip layer around an already gzip-compressed RDS. Remove
  # that outer layer first with `gzip -cd source.RDS.gz > source.RDS`.
  source_object <- readRDS(input_files[[1]])
  for (context in names(source_object)) {
    # Access the inherited S4Vectors DataFrame slot directly. This avoids
    # loading monocle3 merely to dispatch as.data.frame on cell_data_set.
    metadata <- as.data.frame(source_object[[context]]@colData@listData)
    if (!("gene_id" %in% colnames(metadata))) {
      stop("McFaline colData has no gene_id column")
    }
    record_audit(
      "mcfaline2024", context, metadata, "gene_id", c("NA", "NTC", "random")
    )
    assignments <- toupper(as.character(metadata$gene_id))
    single <- !is.na(assignments) & !grepl("[,;+]", assignments)
    write_counts(context, table(assignments[single]), "mcfaline")
  }
} else if (mode == "jiang") {
  suppressPackageStartupMessages(library(SeuratObject))
  accumulated <- list()
  for (input_file in input_files) {
    message("reading ", basename(input_file))
    source_object <- readRDS(input_file)
    metadata <- source_object@meta.data
    if (!all(c("cell_type", "gene") %in% colnames(metadata))) {
      stop("Jiang Seurat metadata lacks cell_type or gene")
    }
    for (context in unique(as.character(metadata$cell_type))) {
      context_metadata <- metadata[as.character(metadata$cell_type) == context, ]
      record_audit("jiang2025", context, context_metadata, "gene", c("NT"))
      assignments <- toupper(as.character(context_metadata$gene))
      counts <- table(assignments)
      if (is.null(accumulated[[context]])) {
        accumulated[[context]] <- counts
      } else {
        genes <- union(names(accumulated[[context]]), names(counts))
        accumulated[[context]] <- setNames(
          vapply(
            genes,
            function(gene) {
              sum(accumulated[[context]][gene], counts[gene], na.rm = TRUE)
            },
            integer(1)
          ),
          genes
        )
      }
    }
    rm(source_object, metadata)
    gc()
  }
  for (context in names(accumulated)) {
    write_counts(context, accumulated[[context]], "jiang")
  }
} else {
  stop("MODE must be jiang or mcfaline")
}

audit <- do.call(rbind, audit_rows)
write.csv(
  audit,
  file.path(output_dir, paste0(mode, "_cell_count_audit.csv")),
  row.names = FALSE,
  quote = FALSE
)
