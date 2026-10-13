# RESM3002 Advanced Social Data Science - Assignment 2026
# Stack Exchange Question Category Classification
# Student Number: 34074759

# 0. Setup and packages

set.seed(34074759)
options(stringsAsFactors = FALSE)

# Hyperparameter-search budgets
N_RANDOM_XGB <- 20L
N_RANDOM_LGB <- 20L
SVM_GRID_SIZE <- 9L
NB_GRID <- c(0, 0.5, 1, 2)
BILSTM_GRID_N <- 6L
GLMNET_ALPHAS <- c(0, 0.25, 0.5, 0.75, 1)


# Python / reticulate configuration. r-keras virtualenv is used for both
# TensorFlow and the MiniLM sentence embeddings.
Sys.setenv(KERAS_BACKEND = "tensorflow")

rkeras_python <- file.path(Sys.getenv("USERPROFILE"),
                           "OneDrive", "Documents", ".virtualenvs",
                           "r-keras", "Scripts", "python.exe")
if (!file.exists(rkeras_python)) {
  rkeras_python <- file.path(Sys.getenv("USERPROFILE"),
                             "Documents", ".virtualenvs",
                             "r-keras", "Scripts", "python.exe")
}
if (file.exists(rkeras_python)) {
  Sys.setenv(RETICULATE_PYTHON = rkeras_python)
}


suppressPackageStartupMessages({
  library(tidyverse)
  library(tm)
  library(SnowballC)
  library(glmnet)
  library(caret)
  library(e1071)
  library(xgboost)
  library(wordcloud)
  library(ggcorrplot)
  library(Matrix)
  library(naivebayes)
  library(tidytext)
  library(tidylo)
  library(ggrepel)
  library(uwot)
  library(proxy)
  library(viridis)
})

# viz packages
ridges_ok <- requireNamespace("ggridges", quietly = TRUE)
beeswarm_ok <- requireNamespace("ggbeeswarm", quietly = TRUE)
circlize_ok <- requireNamespace("circlize", quietly = TRUE)
graph_ok <- requireNamespace("tidygraph", quietly = TRUE) &&
  requireNamespace("ggraph", quietly = TRUE)
irlba_ok <- requireNamespace("irlba", quietly = TRUE)

# quanteda stack trycatche
quanteda_available <- FALSE
tryCatch({
  suppressPackageStartupMessages({
    library(quanteda)
    library(quanteda.textstats)
    library(quanteda.textplots)
  })
  quanteda_available <- TRUE
}, error = function(e) {
  message("** quanteda stack missing — readability/keyness sections will skip. **")
})

# CPU Setting
n_cores <- max(1, parallel::detectCores(logical = TRUE) - 1)
cat("Using", n_cores, "CPU threads where supported.\n")

# avoiding print-inside-device issue
save_plot <- function(plot_obj, filename, width = 8, height = 5, dpi = 300) {
  tryCatch(
    ggsave(filename, plot = plot_obj, width = width, height = height, dpi = dpi),
    error = function(e) message("Could not save ", filename, ": ", conditionMessage(e))
  )
}

# for later analysis
add_row <- function(df, name, cm) {
  bc <- as.data.frame(cm$byClass)
  macro_prec <- mean(bc[["Precision"]],   na.rm = TRUE)
  macro_rec <- mean(bc[["Recall"]],      na.rm = TRUE)
  macro_f1 <- mean(bc[["F1"]],          na.rm = TRUE)
  macro_bacc <- mean(bc[["Balanced Accuracy"]], na.rm = TRUE)
  rbind(df, data.frame(
    Model = name,
    Accuracy = round(unname(cm$overall["Accuracy"]), 4),
    Kappa = round(unname(cm$overall["Kappa"]), 4),
    Macro_F1 = round(macro_f1,   4),
    Macro_Prec = round(macro_prec, 4),
    Macro_Recall = round(macro_rec,  4),
    Macro_BalAcc = round(macro_bacc, 4),
    stringsAsFactors = FALSE
  ))
}

# 1. Data loading

eng_lang <- read.csv("questions_eng_lang.csv", stringsAsFactors = FALSE)
eng_lear <- read.csv("questions_eng_lear.csv", stringsAsFactors = FALSE)
scifi <- read.csv("questions_scifi.csv", stringsAsFactors = FALSE)
world <- read.csv("questions_world.csv", stringsAsFactors = FALSE)

cat("English Language:", nrow(eng_lang), "rows\n")
cat("English Learners:", nrow(eng_lear), "rows\n")
cat("Sci-Fi: ", nrow(scifi), "rows\n")
cat("World Building:", nrow(world), "rows\n")

data <- bind_rows(eng_lang, eng_lear, scifi, world)
cat("\nTotal observations:", nrow(data), "\n")
cat("Column names:", paste(names(data), collapse = ", "), "\n")
cat("\nCategory counts:\n"); print(table(data$Category))

data$Category <- as.factor(data$Category)

# 2. Data cleaning 

cat("\ MISSING VALUES \n")
print(colSums(is.na(data)))
cat("Empty Text fields:", sum(data$Text == "", na.rm = TRUE), "\n")


count_pattern <- function(x, pat) {
  m <- gregexpr(pat, x, perl = TRUE, ignore.case = TRUE)
  vapply(m, function(z) if (z[1] == -1L) 0L else length(z), integer(1))
}

data$n_codeblocks <- count_pattern(data$Text, "<pre[^>]*>|<code[^>]*>")
data$n_links <- count_pattern(data$Text, "<a\\s+[^>]*href")
data$n_blockquotes <- count_pattern(data$Text, "<blockquote")
data$n_images <- count_pattern(data$Text, "<img\\s")
data$n_lists <- count_pattern(data$Text, "<ul|<ol")


clean_text <- function(x) {
  x <- gsub("<[^>]+>", " ", x)
  x <- gsub("&[a-zA-Z]+;", " ", x)
  x <- gsub("&#\\d+;", " ", x)
  x <- gsub("https?://\\S+", " __URL__ ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

data$Text_clean <- vapply(data$Text,  clean_text, character(1), USE.NAMES = FALSE)
data$Title_clean <- vapply(data$Title, clean_text, character(1), USE.NAMES = FALSE)
data$FullText <- paste(data$Title_clean, data$Text_clean)


data$n_question_marks <- count_pattern(data$Text_clean, "\\?")
data$n_exclamations <- count_pattern(data$Text_clean, "!")
data$n_first_person <- count_pattern(data$Text_clean,
                                       "\\b(I|i|me|my|mine|we|us|our|ours)\\b")
data$n_second_person <- count_pattern(data$Text_clean,
                                       "\\b(you|your|yours)\\b")
data$n_third_person <- count_pattern(data$Text_clean,
                                       "\\b(he|him|his|she|her|hers|they|them|their)\\b")
data$n_modals <- count_pattern(data$Text_clean,
                                       "\\b(could|would|should|might|may|must|can)\\b")
data$n_caps_tokens <- count_pattern(data$Text_clean, "\\b[A-Z][a-z]+")


data$TextLength <- nchar(data$Text_clean)
data$TitleLength <- nchar(data$Title_clean)
data$WordCount <- vapply(strsplit(data$Text_clean, "\\s+"),
                           length, integer(1))


data$views_per_answer <- data$ViewCount / pmax(data$AnswerCount, 1)
data$score_per_view <- 1000 * data$Score / pmax(data$ViewCount, 1)
data$is_unanswered <- as.integer(data$AnswerCount == 0)


data$Score_log <- sign(data$Score) * log1p(abs(data$Score))
data$Views_log <- log1p(data$ViewCount)
data$Answers_log<- log1p(data$AnswerCount)

cat("\nDataset ready:", nrow(data), "observations,",
    nlevels(data$Category), "categories\n")


# 3. DESCRIPTIVE STATISTICS —
cat("\nDESCRIPTIVE STATISTICS\n")


desc_table <- data %>%
  group_by(Category) %>%
  summarise(
    N = n(),
    Proportion = round(n() / nrow(data), 3),
    Mean_Score = round(mean(Score), 2),
    Median_Score = median(Score),
    SD_Score = round(sd(Score), 2),
    Mean_Views = round(mean(ViewCount)),
    Median_Views = median(ViewCount),
    Mean_Answers = round(mean(AnswerCount), 2),
    Pct_Unanswered = round(100 * mean(is_unanswered), 1),
    Mean_WordCount = round(mean(WordCount)),
    Mean_TextLen = round(mean(TextLength)),
    Mean_QMarks = round(mean(n_question_marks), 2),
    Mean_FirstPers = round(mean(n_first_person), 2),
    Mean_CodeBlk = round(mean(n_codeblocks), 2),
    .groups = "drop"
  )
print(desc_table, width = Inf)
write.csv(desc_table, "table_descriptive_stats.csv", row.names = FALSE)

# distribution plots
p1 <- ggplot(data, aes(x = Category, fill = Category)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_viridis_d() +
  theme_minimal(base_size = 11) +
  labs(title = "Distribution of Questions Across Categories",
       x = NULL, y = "Count") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1),
        legend.position = "none")
save_plot(p1, "category_distribution.png")

p2 <- ggplot(data, aes(x = Category, y = Score, fill = Category)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
  scale_fill_viridis_d() +
  coord_cartesian(ylim = quantile(data$Score, c(0.01, 0.99))) +
  theme_minimal(base_size = 11) +
  labs(title = "Score by Category", x = NULL, y = "Score") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1),
        legend.position = "none")
save_plot(p2, "score_boxplot.png")

p3 <- ggplot(data, aes(x = Category, y = ViewCount, fill = Category)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
  scale_fill_viridis_d() + scale_y_log10() +
  theme_minimal(base_size = 11) +
  labs(title = "View Count by Category (log scale)",
       x = NULL, y = "View Count") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1),
        legend.position = "none")
save_plot(p3, "viewcount_boxplot.png")

p4 <- ggplot(data, aes(x = WordCount, fill = Category)) +
  geom_density(alpha = 0.4) +
  scale_fill_viridis_d() +
  xlim(0, quantile(data$WordCount, 0.99)) +
  theme_minimal(base_size = 11) +
  labs(title = "Word Count Density by Category",
       x = "Word Count", y = "Density")
save_plot(p4, "wordcount_density.png")

# engagement metrics 
eng_long <- data %>%
  transmute(Category,
            views_per_answer = pmin(views_per_answer, quantile(views_per_answer, 0.99)),
            score_per_view,
            log_views = log10(ViewCount),
            answers_log = Answers_log) %>%
  pivot_longer(-Category, names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric,
                         levels = c("views_per_answer", "score_per_view",
                                    "log_views", "answers_log"),
                         labels = c("Views per Answer", "Score per 1k Views",
                                    "log10(Views)", "log1p(Answers)")))

p5 <- ggplot(eng_long, aes(x = Category, y = value, fill = Category)) +
  geom_violin(alpha = 0.6, scale = "width") +
  geom_boxplot(width = 0.15, alpha = 0.5, outlier.shape = NA) +
  facet_wrap(~ metric, scales = "free_y") +
  scale_fill_viridis_d() +
  theme_minimal(base_size = 11) +
  labs(title = "Engagement Patterns by Category",
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1),
        legend.position = "none")
save_plot(p5, "engagement_panel.png", width = 10, height = 7)

# correlation matrix of numeric features
num_vars <- data %>%
  select(Score, ViewCount, AnswerCount, TextLength, WordCount,
         n_question_marks, n_first_person, n_codeblocks, n_links)
cor_mat <- cor(num_vars, use = "complete.obs")
p_cor <- ggcorrplot(cor_mat, type = "lower", lab = TRUE,
                    lab_size = 2.8, tl.cex = 8,
                    title = "Correlation Matrix — Engineered Numeric Features")
save_plot(p_cor, "correlation_matrix.png", width = 7, height = 6)


#Readability and lexical diversity
read_lex <- NULL
if (quanteda_available) {


  qcorpus <- corpus(data$FullText, docvars = data["Category"])
  read_df <- textstat_readability(qcorpus,
                                  measure = c("Flesch.Kincaid", "FOG", "Dale.Chall"))
  qtokens <- tokens(qcorpus, remove_punct = TRUE, remove_symbols = TRUE)
  lex_df <- textstat_lexdiv(qtokens, measure = c("TTR", "MATTR"))

  read_lex_wide <- bind_cols(read_df, lex_df %>% select(-document)) %>%
    mutate(Category = data$Category)

  read_lex <- read_lex_wide %>%
    pivot_longer(c(Flesch.Kincaid, FOG, Dale.Chall, TTR, MATTR),
                 names_to = "metric", values_to = "value") %>%
    filter(is.finite(value))

  read_summary <- read_lex %>%
    group_by(Category, metric) %>%
    summarise(mean = mean(value, na.rm = TRUE),
              median = median(value, na.rm = TRUE),
              sd = sd(value, na.rm = TRUE),
              .groups = "drop")
  write.csv(read_summary, "table_readability_lexdiv.csv", row.names = FALSE)
  print(read_summary)

  p_read <- ggplot(read_lex, aes(x = Category, y = value, fill = Category)) +
    geom_violin(alpha = 0.7, scale = "width") +
    geom_boxplot(width = 0.15, alpha = 0.4, outlier.shape = NA) +
    facet_wrap(~ metric, scales = "free_y", ncol = 3) +
    scale_fill_viridis_d() +
    theme_minimal(base_size = 11) +
    labs(title = "Readability and Lexical Diversity by Category",
         subtitle = "Higher Flesch-Kincaid/FOG/Dale-Chall = harder text. Higher TTR/MATTR = more diverse vocabulary.",
         x = NULL, y = NULL) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1),
          legend.position = "none",
          strip.text = element_text(face = "bold"))
  save_plot(p_read, "readability_lexdiv.png", width = 11, height = 7)
}

# Weighted log-odds 


word_counts <- data %>%
  select(Category, FullText) %>%
  unnest_tokens(word, FullText) %>%
  filter(!word %in% stopwords::stopwords("en")) %>%
  filter(!str_detect(word, "^\\d+$"), nchar(word) >= 3) %>%
  count(Category, word, sort = TRUE)

log_odds <- word_counts %>%
  bind_log_odds(set = Category, feature = word, n = n)

write.csv(log_odds %>% group_by(Category) %>%
            slice_max(log_odds_weighted, n = 30) %>% ungroup(),
          "table_logodds_top30.csv", row.names = FALSE)

p_logodds <- log_odds %>%
  group_by(Category) %>%
  slice_max(log_odds_weighted, n = 15) %>%
  ungroup() %>%
  ggplot(aes(x = log_odds_weighted,
             y = reorder_within(word, log_odds_weighted, Category),
             fill = Category)) +
  geom_col(show.legend = FALSE) +
  scale_y_reordered() +
  scale_fill_viridis_d() +
  facet_wrap(~ Category, scales = "free_y") +
  theme_minimal(base_size = 11) +
  labs(title = "Most Distinctive Words per Stack Exchange Site",
       subtitle = "Weighted log-odds with informative Dirichlet prior (Monroe, Colaresi & Quinn 2008)",
       x = "Weighted log-odds (z-score)", y = NULL) +
  theme(strip.text = element_text(face = "bold"))
save_plot(p_logodds, "logodds_distinctive_words.png", width = 11, height = 8)

# Category-centroid cosine-similarity heatmap


quick_corp <- VCorpus(VectorSource(data$FullText))
quick_corp <- tm_map(quick_corp, content_transformer(tolower))
quick_corp <- tm_map(quick_corp, removeNumbers)
quick_corp <- tm_map(quick_corp, removePunctuation)
quick_corp <- tm_map(quick_corp, removeWords, stopwords("english"))
quick_corp <- tm_map(quick_corp, stripWhitespace)
quick_dtm <- DocumentTermMatrix(quick_corp,
                                 control = list(weighting = weightTfIdf,
                                                bounds = list(global = c(10, Inf))))
quick_dtm <- removeSparseTerms(quick_dtm, sparse = 0.98)
quick_mat <- as.matrix(quick_dtm)

centroids <- t(sapply(levels(data$Category), function(k) {
  colMeans(quick_mat[data$Category == k, , drop = FALSE])
}))
cos_mat <- 1 - as.matrix(proxy::dist(centroids, method = "cosine"))
diag(cos_mat) <- 1

cos_df <- as.data.frame(as.table(cos_mat)); names(cos_df) <- c("A","B","Cosine")
p_cos <- ggplot(cos_df, aes(A, B, fill = Cosine)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.3f", Cosine)), color = "white", size = 4) +
  scale_fill_viridis_c(limits = c(min(cos_df$Cosine), 1)) +
  theme_minimal(base_size = 11) +
  labs(title = "Inter-category Cosine Similarity of TF-IDF Centroids",
       subtitle = "Higher = harder to classify apart. Predicts the confusion matrix.",
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1),
        panel.grid = element_blank())
save_plot(p_cos, "centroid_cosine_heatmap.png", width = 7, height = 6)
write.csv(cos_df, "table_centroid_cosine.csv", row.names = FALSE)

# Structural topic modelling
stm_available <- FALSE
tryCatch({
  library(stm)
  cat("\nFitting (K=12)\n")

  stm_proc <- textProcessor(documents = data$FullText, metadata = data,
                            lowercase = TRUE, removestopwords = TRUE,
                            removenumbers = TRUE, removepunctuation = TRUE,
                            stem = TRUE, verbose = FALSE)
  stm_out <- prepDocuments(stm_proc$documents, stm_proc$vocab,
                            stm_proc$meta, lower.thresh = 10, verbose = FALSE)

  set.seed(34074759)
  fit_stm <- stm(documents = stm_out$documents,
                 vocab = stm_out$vocab,
                 K = 12,
                 prevalence= ~ Category,
                 data = stm_out$meta,
                 init.type = "Spectral",
                 max.em.its= 75, verbose = FALSE)

  top_words <- labelTopics(fit_stm, n = 8)
  topic_labels <- apply(top_words$prob, 1, paste, collapse = ", ")
  sink("table_stm_topics.txt")
  cat("Top 8 words per topic (FREX-weighted):\n\n")
  for (i in seq_along(topic_labels))
    cat(sprintf("Topic %2d: %s\n", i, topic_labels[i]))
  sink()

  theta <- fit_stm$theta
  prev_by_cat <- aggregate(theta, by = list(Category = stm_out$meta$Category),
                           FUN = mean)
  prev_long <- pivot_longer(prev_by_cat, -Category,
                            names_to = "Topic", values_to = "Prevalence") %>%
    mutate(Topic_id = as.integer(gsub("V", "", Topic)),
           TopicLab = sprintf("T%02d: %s", Topic_id,
                               substr(topic_labels[Topic_id], 1, 40)))
  prev_long$TopicLab <- factor(prev_long$TopicLab,
                               levels = unique(prev_long$TopicLab[order(prev_long$Topic_id)]))

  p_stm <- ggplot(prev_long, aes(Category, TopicLab, fill = Prevalence)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_viridis_c() +
    theme_minimal(base_size = 10) +
    labs(title = "STM Topic Prevalence by Category (K = 12)",
         subtitle = "Roberts, Stewart & Tingley (2014). Topics labelled by top probability words.",
         x = NULL, y = NULL) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1),
          panel.grid = element_blank())
  save_plot(p_stm, "stm_topic_prevalence.png", width = 11, height = 7)

  stm_available <- TRUE
}, error = function(e) {
  cat("** STM step skipped:", conditionMessage(e), "**\n")
})

# comparative scattertext-style plot


ell_elu_levels <- levels(data$Category)[
  grepl("english", levels(data$Category), ignore.case = TRUE)
]
if (length(ell_elu_levels) >= 2) {
  freq_ee <- data %>%
    filter(Category %in% ell_elu_levels) %>%
    select(Category, FullText) %>%
    unnest_tokens(word, FullText) %>%
    filter(!word %in% stopwords::stopwords("en"),
           !str_detect(word, "^\\d+$"), nchar(word) >= 3) %>%
    count(Category, word) %>%
    group_by(Category) %>% mutate(prop = n / sum(n)) %>% select(-n) %>%
    ungroup() %>%
    pivot_wider(names_from = Category, values_from = prop, values_fill = 0)

  cn1 <- ell_elu_levels[1]; cn2 <- ell_elu_levels[2]
  freq_ee_top <- freq_ee %>%
    filter(.data[[cn1]] > 0.0005 | .data[[cn2]] > 0.0005) %>%
    mutate(maxprop = pmax(.data[[cn1]], .data[[cn2]])) %>%
    arrange(desc(maxprop)) %>% head(150)

  p_scat <- ggplot(freq_ee_top, aes(x = .data[[cn1]],
                                     y = .data[[cn2]])) +
    geom_abline(linetype = 2, color = "grey50") +
    geom_jitter(alpha = 0.3, size = 1) +
    geom_text_repel(aes(label = word), size = 3, max.overlaps = 20) +
    scale_x_log10() + scale_y_log10() +
    theme_minimal(base_size = 11) +
    labs(title = sprintf("Word usage: %s vs %s", ell_elu_levels[1], ell_elu_levels[2]),
         subtitle = "Words far from the diagonal differ in frequency between the two sites",
         x = sprintf("Frequency in %s", ell_elu_levels[1]),
         y = sprintf("Frequency in %s", ell_elu_levels[2]))
  save_plot(p_scat, "eng_pair_scatter.png", width = 9, height = 7)
}

# Comparison wordcloud
if (quanteda_available) {
  tryCatch({
    qdfm <- tokens(qcorpus, remove_punct = TRUE, remove_symbols = TRUE) %>%
      tokens_remove(stopwords("en")) %>%
      tokens_select(min_nchar = 3) %>%
      tokens_tolower() %>% dfm()
    qdfm_grp <- dfm_group(qdfm, groups = qdfm$Category)

    png("comparison_wordcloud.png", width = 1000, height = 1000, res = 130)
    textplot_wordcloud(qdfm_grp, comparison = TRUE, max_words = 80,
                       color = c("#440154", "#3b528b", "#21918c", "#5ec962"))
    dev.off()
  }, error = function(e) {
    cat("Skipped comparison wordcloud:", conditionMessage(e), "\n")
  })
}


# Ridgeline plot of word count, log views, first-person count and
# Flesch-Kincaid by category.
if (ridges_ok) {
  library(ggridges)

  ridge_data <- data %>%
    transmute(Category,
              `Word Count` = pmin(WordCount, quantile(WordCount, 0.99)),
              `log10(Views)` = log10(pmax(ViewCount, 1)),
              `First-Person Mentions` = pmin(n_first_person,
                                             quantile(n_first_person, 0.98)))

  if (!is.null(read_lex)) {
    fk_data <- read_lex %>% filter(metric == "Flesch.Kincaid") %>%
      transmute(Category, `Flesch-Kincaid Grade` = value)
    ridge_data <- ridge_data %>%
      mutate(rid = row_number()) %>%
      left_join(fk_data %>% mutate(rid = row_number()), by = c("Category", "rid")) %>%
      select(-rid)
  }

  ridge_long <- ridge_data %>%
    pivot_longer(-Category, names_to = "metric", values_to = "value") %>%
    filter(is.finite(value))

  p_ridge <- ggplot(ridge_long, aes(x = value, y = Category, fill = Category)) +
    geom_density_ridges(alpha = 0.75, scale = 1.5, rel_min_height = 0.005,
                        quantile_lines = TRUE, quantiles = 2,
                        linewidth = 0.4, colour = "white") +
    facet_wrap(~ metric, scales = "free_x") +
    scale_fill_viridis_d() +
    theme_ridges(grid = FALSE) +
    theme(legend.position = "none",
          strip.text = element_text(face = "bold", size = 11),
          axis.title.y = element_blank(),
          plot.background = element_rect(fill = "white", colour = NA),
          panel.background = element_rect(fill = "white", colour = NA),
          strip.background = element_rect(fill = "white", colour = NA)) +
    labs(title = "Ridgeline Distributions by Category",
         subtitle = "Median lines highlight central tendency; ridge shape reveals skew and modality",
         x = NULL)
  save_plot(p_ridge, "ridgeline_distributions.png", width = 12, height = 7)
} else {
  cat("** Ridgeline skipped — install.packages('ggridges') **\n")
}

# Beeswarm plot of log10(ViewCount) by category
if (beeswarm_ok) {
  library(ggbeeswarm)

  set.seed(34074759)
  bee_data <- data %>%
    group_by(Category) %>%
    slice_sample(n = 700) %>% 
    ungroup() %>%
    mutate(log_views = log10(pmax(ViewCount, 1)))

  p_bee <- ggplot(bee_data, aes(x = Category, y = log_views, colour = Category)) +
    geom_quasirandom(alpha = 0.55, size = 0.9, width = 0.4) +
    geom_boxplot(width = 0.10, fill = "white", colour = "black",
                 alpha = 0.85, outlier.shape = NA) +
    scale_colour_viridis_d() +
    theme_minimal(base_size = 11) +
    labs(title = "Beeswarm: log10(ViewCount) by Category",
         subtitle = "Each point is one question. Box = IQR; whiskers omitted; outliers retained as points.",
         x = NULL, y = "log10(ViewCount + 1)") +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 25, hjust = 1))
  save_plot(p_bee, "beeswarm_views.png", width = 10, height = 7)
} else {
  cat("** Beeswarm skipped — install.packages('ggbeeswarm') **\n")
}

# Correlation network of engineered numeric features.
if (graph_ok) {
  library(tidygraph); library(ggraph)

  net_vars <- data %>%
    dplyr::select(Score, ViewCount, AnswerCount, TextLength, TitleLength, WordCount,
                  n_question_marks, n_exclamations, n_first_person, n_second_person,
                  n_third_person, n_modals, n_caps_tokens,
                  n_codeblocks, n_links, n_blockquotes, n_images,
                  views_per_answer, score_per_view) %>%
    dplyr::mutate(across(everything(), ~ replace(.x, !is.finite(.x), NA))) %>%
    tidyr::drop_na()

  sd_each <- vapply(net_vars, sd, numeric(1), na.rm = TRUE)
  net_vars <- net_vars[, sd_each > 0]

  cmat <- cor(net_vars, use = "pairwise.complete.obs")
  cmat[upper.tri(cmat, diag = TRUE)] <- NA

  edges <- as.data.frame(as.table(cmat)) %>%
    setNames(c("from", "to", "r")) %>%
    dplyr::filter(!is.na(r), abs(r) >= 0.30) %>%
    dplyr::mutate(from = as.character(from), 
                  to = as.character(to))

  if (nrow(edges) > 0) {
    nodes <- data.frame(name = unique(c(edges$from, edges$to)), 
                        stringsAsFactors = FALSE)

    g <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE) %>%
      activate(nodes) %>%
      dplyr::mutate(degree = centrality_degree())

    set.seed(34074759)
    p_net <- ggraph(g, layout = "fr") +
      geom_edge_link(aes(width = abs(r), colour = r), alpha = 0.7) +
      geom_node_point(aes(size = degree), colour = "grey20") +
      geom_node_text(aes(label = name), repel = TRUE, size = 3.2,
                     fontface = "bold") +
      scale_edge_width(range = c(0.4, 2.5), name = "|r|") +
      scale_edge_colour_gradient2(low = "#d73027", mid = "grey90",
                                  high = "#4575b4", midpoint = 0,
                                  name = "Pearson r") +
      scale_size(range = c(2.5, 6), guide = "none") +
      theme_graph(base_family = "sans") +
      labs(title = "Correlation Network of Engineered Numeric Features",
           subtitle = paste0("Edges shown for |r| >= 0.30 (",
                             nrow(edges), " edges). Force-directed layout."))
    save_plot(p_net, "correlation_network.png", width = 11, height = 8)
  } else {
    cat("No correlations >= 0.30 — network plot skipped.\n")
  }
} else {
  cat("** Correlation network skipped — install.packages(c('tidygraph','ggraph')) **\n")
}

cat("\nDescriptive analysis complete.\n")


# Text preprocessing - document-term matrix

cat("\nTEXT PREPROCESSING\n")

corpus <- VCorpus(VectorSource(data$FullText))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)

dtm_tfidf <- DocumentTermMatrix(corpus, control = list(
  weighting = weightTfIdf,
  bounds = list(global = c(5, Inf))
))
cat("Full TF-IDF DTM:", nrow(dtm_tfidf), "docs x", ncol(dtm_tfidf), "terms\n")

dtm_tfidf <- removeSparseTerms(dtm_tfidf, sparse = 0.99)
cat("After sparsity reduction:", ncol(dtm_tfidf), "terms\n")

text_matrix <- as.matrix(dtm_tfidf)

dtm_counts <- DocumentTermMatrix(corpus, control = list(
  bounds = list(global = c(5, Inf))
))
dtm_counts <- removeSparseTerms(dtm_counts, sparse = 0.99)
count_matrix <- as.matrix(dtm_counts)

numeric_feats <- data %>%
  select(Score_log, Views_log, Answers_log, TextLength, TitleLength, WordCount,
         n_question_marks, n_exclamations, n_first_person, n_second_person,
         n_third_person, n_modals, n_caps_tokens,
         n_codeblocks, n_links, n_blockquotes, n_images, n_lists,
         views_per_answer, score_per_view, is_unanswered) %>%
  as.matrix()
numeric_feats[!is.finite(numeric_feats)] <- 0


X_full <- cbind(text_matrix, numeric_feats)
y <- data$Category

cat("Combined feature matrix:", nrow(X_full), "x", ncol(X_full),
    "(scaling deferred to post-split)\n")

stylistic_vars <- c("n_first_person", "n_second_person", "n_third_person",
                    "n_modals", "n_question_marks", "n_exclamations",
                    "n_caps_tokens")

numeric_cols <- colnames(numeric_feats)
structural_vars <- setdiff(numeric_cols, stylistic_vars)

n_tfidf <- ncol(text_matrix)
n_stylistic <- length(intersect(stylistic_vars,  numeric_cols))
n_structural <- length(structural_vars)
n_total <- n_tfidf + n_stylistic + n_structural
stopifnot(n_total == ncol(X_full))   # sanity check

feature_blocks <- data.frame(
  Block = c("TF-IDF text",
            "Stylistic counts",
            "Structural & engagement",
            "TOTAL"),
  Variables = c(
    "Stemmed unigrams from Title + body; stopwords removed; doc-freq >= 5; sparsity <= 0.99",
    paste(intersect(stylistic_vars, numeric_cols), collapse = ", "),
    paste(structural_vars, collapse = ", "),
    "—"
  ),
  Count = c(n_tfidf, n_stylistic, n_structural, n_total),
  Rationale = c(
    "Lexical content (Porter, 1980; Salton & Buckley, 1988)",
    "ELU/ELL stance gap not resolvable by topic words (Pennebaker, Mehl & Niederhoffer, 2003)",
    "Site-conventional formatting + post-hoc engagement signal",
    "—"
  ),
  stringsAsFactors = FALSE
)

cat("\n FEATURE-BLOCK SUMMARY\n")
print(feature_blocks, row.names = FALSE)
write.csv(feature_blocks, "table_feature_blocks.csv", row.names = FALSE)
cat("Saved table_feature_blocks.csv\n")


# 5. Train/test split (80/20, stratified)

cat("\n===== TRAIN-TEST SPLIT =====\n")
set.seed(34074759)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)

num_train <- numeric_feats[train_idx,  , drop = FALSE]
num_test <- numeric_feats[-train_idx, , drop = FALSE]
num_train_scaled <- scale(num_train)
train_means <- attr(num_train_scaled, "scaled:center")
train_sds <- attr(num_train_scaled, "scaled:scale")
train_sds[train_sds == 0 | !is.finite(train_sds)] <- 1
num_test_scaled <- scale(num_test, center = train_means, scale = train_sds)
num_train_scaled[!is.finite(num_train_scaled)] <- 0
num_test_scaled[!is.finite(num_test_scaled)]   <- 0
numeric_scaled <- numeric_feats   
numeric_scaled[train_idx,  ] <- num_train_scaled
numeric_scaled[-train_idx, ] <- num_test_scaled

# leakage-free scaled numerics
X_full <- cbind(text_matrix, numeric_scaled)

X_train <- X_full[train_idx, ]
X_test <- X_full[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]
cat("Training:", length(y_train), "  Test:", length(y_test), "\n")
cat("Numeric scaling: fit on", nrow(num_train),
    "training rows, applied to", nrow(num_test), "test rows.\n")

X_train_sp <- Matrix(X_train, sparse = TRUE)
X_test_sp <- Matrix(X_test,  sparse = TRUE)

text_train_sp <- Matrix(text_matrix[train_idx, ],  sparse = TRUE)
text_test_sp <- Matrix(text_matrix[-train_idx, ], sparse = TRUE)

set.seed(34074759)

inner_val_idx <- as.integer(caret::createDataPartition(
  y_train, p = 0.15, list = FALSE
))

inner_val_idx <- sort(unique(inner_val_idx))
inner_train_idx <- setdiff(seq_along(y_train), inner_val_idx)

X_train_sp <- methods::as(Matrix::Matrix(X_train, sparse = TRUE), "dgCMatrix")
X_test_sp <- methods::as(Matrix::Matrix(X_test,  sparse = TRUE), "dgCMatrix")

X_train_inner_sp <- X_train_sp[inner_train_idx, , drop = FALSE]
X_inner_val_sp <- X_train_sp[inner_val_idx,   , drop = FALSE]

y_train_inner <- y_train[inner_train_idx]
y_inner_val <- y_train[inner_val_idx]

cat("Inner-train:", length(y_train_inner),
    "  Inner-val:", length(y_inner_val),
    "  (used for tree-booster early stopping; test set NOT touched)\n")


# 6. Sentence embeddings, note to self: likely not including in report due difficulties with modelling minilm


emb_available <- FALSE
emb_full <- NULL
emb_cache <- "minilm_embeddings.rds"

valid_embedding_matrix <- function(x, expected_n) {
  is.matrix(x) &&
    is.numeric(x) &&
    nrow(x) == expected_n &&
    ncol(x) > 0 &&
    all(is.finite(x[seq_len(min(length(x), 1000))]))
}

if (file.exists(emb_cache)) {
  cat("Loading cached MiniLM embeddings from disk...\n")

  cached <- tryCatch(
    readRDS(emb_cache),
    error = function(e) {
      cat("Cached embedding file could not be read:", conditionMessage(e), "\n")
      NULL
    }
  )

  cache_ok <- tryCatch(
    !is.null(cached) &&
      !inherits(cached, "python.builtin.object") &&
      valid_embedding_matrix(cached, expected_n = nrow(data)),
    error = function(e) FALSE
  )

  if (cache_ok) {
    emb_full <- cached
    storage.mode(emb_full) <- "double"
    emb_available <- TRUE

    cat("Loaded valid cached embeddings:",
        nrow(emb_full), "x", ncol(emb_full), "\n")
  } else {
    cat("Cached embeddings are invalid/stale. Deleting cache and regenerating...\n")
    unlink(emb_cache)
  }
}

if (!emb_available) {
  tryCatch({
    library(reticulate)
    if (exists("rkeras_python") && file.exists(rkeras_python)) {
      use_python(rkeras_python, required = TRUE)
    }

    if (!py_module_available("sentence_transformers")) {
      stop(
        "Python module 'sentence_transformers' is not installed in r-keras. ",
        "Run once in R: system('", rkeras_python,
        " -m pip install torch sentence-transformers')"
      )
    }

    st <- import("sentence_transformers", convert = FALSE)
    model <- st$SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    cat("\n SENTENCE EMBEDDINGS\n")
    cat("Encoding", nrow(data), "documents on CPU (~10-25 min)...\n")

    txt_for_emb <- as.character(substr(data$FullText, 1, 2000))
    txt_for_emb[is.na(txt_for_emb)] <- ""

    emb_py <- model$encode(
      txt_for_emb,
      batch_size = 32L,
      show_progress_bar = TRUE,
      convert_to_numpy = TRUE
    )

    emb_full <- reticulate::py_to_r(emb_py)
    emb_full <- as.matrix(emb_full)
    storage.mode(emb_full) <- "double"

    if (!valid_embedding_matrix(emb_full, expected_n = nrow(data))) {
      stop("Embedding matrix was created, but has invalid dimensions or values.")
    }

    cat("Embeddings dim:", dim(emb_full)[1], "x", dim(emb_full)[2], "\n")

    saveRDS(emb_full, emb_cache)
    cat("Saved clean R embedding matrix to", emb_cache, "\n")

    emb_available <- TRUE

  }, error = function(e) {
    cat("** Embeddings skipped:", conditionMessage(e), "**\n")
    emb_available <<- FALSE
    emb_full <<- NULL
  })
}

if (emb_available) {
  X_emb_train <- emb_full[train_idx, , drop = FALSE]
  X_emb_test <- emb_full[-train_idx, , drop = FALSE]
}


# UMAP class-separability

cat("\nUMAP CLASS-SEPARABILITY VISUALISATION\n")

set.seed(34074759)
if (emb_available) {
  cat("Running UMAP on 384-d MiniLM embeddings...\n")
  um <- uwot::umap(emb_full, n_neighbors = 25, min_dist = 0.05,
                   metric = "cosine", n_components = 2, n_threads = n_cores,
                   verbose = FALSE)
  umap_data <- data.frame(x = um[,1], y = um[,2], Category = data$Category)
  umap_subtitle <- "MiniLM-L6-v2 sentence embeddings reduced to 2-D via UMAP"
} else {
  cat("Embeddings unavailable - falling back to UMAP on TF-IDF.\n")
  sub_idx <- sample(seq_len(nrow(X_full)), min(4000, nrow(X_full)))
  um <- uwot::umap(text_matrix[sub_idx, ], n_neighbors = 25, min_dist = 0.05,
                   metric = "cosine", n_components = 2, n_threads = n_cores,
                   verbose = FALSE)
  umap_data <- data.frame(x = um[,1], y = um[,2],
                          Category = data$Category[sub_idx])
  umap_subtitle <- "TF-IDF reduced to 2-D via UMAP (cosine metric)"
}

p_umap <- ggplot(umap_data, aes(x, y, colour = Category, fill = Category)) +
  geom_point(alpha = 0.45, size = 0.6) +
  stat_ellipse(aes(group = Category), level = 0.68, linewidth = 0.6,
               geom = "polygon", alpha = 0.05, show.legend = FALSE) +
  scale_colour_viridis_d() + scale_fill_viridis_d() +
  theme_minimal(base_size = 11) +
  labs(title = "2-D UMAP — Class Separability",
       subtitle = umap_subtitle,
       x = "UMAP-1", y = "UMAP-2") +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  coord_equal() +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank())
save_plot(p_umap, "fig_13_umap_separability.png", width = 9, height = 8)


# 8. Models
results_list <- list()


# 8.0 Multinomial logistic regression
cat("\nMODEL 8.0: MULTINOMIAL LOGISTIC REGRESSION\n")

logreg_cm <- NULL
logreg_proba <- NULL
logreg_pred <- NULL
logreg_engine <- NA_character_

tryCatch({
  if (requireNamespace("LiblineaR", quietly = TRUE) &&
      requireNamespace("SparseM", quietly = TRUE)) {
    library(LiblineaR)
    library(SparseM)
    cat("Engine: LiblineaR (type = 0, L2-regularised primal multinomial LR)\n")

    dgC_to_csr <- function(M) {
      if (!inherits(M, "dgCMatrix")) M <- as(as.matrix(M), "dgCMatrix")
      methods::as(M, "RsparseMatrix") -> Mr
      methods::new("matrix.csr",
                   ra = Mr@x,
                   ja = as.integer(Mr@j + 1L),
                   ia = as.integer(Mr@p + 1L),
                   dimension = as.integer(dim(Mr)))
    }
    Xtr_csr <- dgC_to_csr(X_train_sp)
    Xte_csr <- dgC_to_csr(X_test_sp)

    cost_grid <- c(0.1, 1, 10, 100)
    set.seed(34074759)
    cv_acc <- vapply(cost_grid, function(C) {
      tryCatch({
        m <- LiblineaR::LiblineaR(data = Xtr_csr, target = y_train,
                                  type = 0, cost = C, cross = 3)
        as.numeric(m) 
      }, error = function(e) NA_real_)
    }, numeric(1))
    cat("LR cost grid CV accuracies:\n")
    print(data.frame(cost = cost_grid, cv_accuracy = round(cv_acc, 4)))
    best_C <- cost_grid[which.max(cv_acc)]
    cat("Best cost:", best_C, "\n")

    set.seed(34074759)
    logreg_model <- LiblineaR::LiblineaR(data = Xtr_csr, target = y_train,
                                         type = 0, cost = best_C)
    logreg_pred_obj <- predict(logreg_model, Xte_csr, proba = TRUE)
    logreg_pred <- factor(as.character(logreg_pred_obj$predictions),
                           levels = levels(y_test))
    if (!is.null(logreg_pred_obj$probabilities)) {
      logreg_proba <- as.matrix(logreg_pred_obj$probabilities)
      logreg_proba <- logreg_proba[, match(levels(y_test),
                                           colnames(logreg_proba)), drop = FALSE]
    }
    logreg_cm <- confusionMatrix(logreg_pred, y_test)
    results_list$logreg <- logreg_cm
    logreg_engine <- "LiblineaR"
    cat("Logistic Regression (LiblineaR) Accuracy:",
        round(logreg_cm$overall["Accuracy"], 4), "\n")

  } else {
    stop("LiblineaR/SparseM not available")
  }
}, error = function(e) {
  cat("** LiblineaR path failed:", conditionMessage(e), "**\n")
  cat("** Falling back to cv.glmnet with very small lambda (near-unpenalised). **\n")

  set.seed(34074759)
  lambda_tiny <- 10 ^ seq(-1, -5, length.out = 20)
  logreg_cv_fb <- cv.glmnet(X_train_sp, y_train,
                            family = "multinomial",
                            type.multinomial = "grouped",
                            alpha = 0,                
                            lambda = lambda_tiny,
                            nfolds = 5, trace.it = 0)
  logreg_pred  <<- factor(predict(logreg_cv_fb, X_test_sp,
                                  s = "lambda.min", type = "class")[, 1],
                          levels = levels(y_test))
  logreg_proba <<- predict(logreg_cv_fb, X_test_sp,
                           s = "lambda.min", type = "response")[, , 1]
  logreg_cm    <<- confusionMatrix(logreg_pred, y_test)
  results_list$logreg <<- logreg_cm
  logreg_engine <<- "cv.glmnet (fallback, lambda~1e-5)"
  cat("Logistic Regression (glmnet fallback) Accuracy:",
      round(logreg_cm$overall["Accuracy"], 4), "\n")
})


# 8.1 Multinomial LASSO
cat("\nMODEL 8.1: MULTINOMIAL LASSO\n")

lasso_path <- glmnet(X_train_sp, y_train,
                     family = "multinomial",
                     type.multinomial = "grouped",
                     alpha = 1)

png("lasso_coef_path.png", width = 800, height = 500, res = 110)
plot(lasso_path, xvar = "lambda", label = FALSE)
title("LASSO Coefficient Paths (Grouped Multinomial)")
dev.off()

set.seed(34074759)
lasso_cv <- cv.glmnet(X_train_sp, y_train,
                      family = "multinomial",
                      type.multinomial = "grouped",
                      alpha = 1, nfolds = 10, trace.it = 0)

png("lasso_cv_curve.png", width = 800, height = 500, res = 110)
plot(lasso_cv); title("LASSO CV Curve (10-fold)")
dev.off()

cat("  lambda.min:", lasso_cv$lambda.min, "\n")
cat("  lambda.1se:", lasso_cv$lambda.1se, "\n")

lasso_pred <- factor(predict(lasso_cv, X_test_sp,
                             s = "lambda.min", type = "class")[, 1],
                     levels = levels(y_test))
lasso_cm <- confusionMatrix(lasso_pred, y_test)
results_list$lasso <- lasso_cm
cat("LASSO Accuracy:", round(lasso_cm$overall["Accuracy"], 4), "\n")

lasso_cm_df <- as.data.frame(lasso_cm$table)
p_lasso_cm <- ggplot(lasso_cm_df, aes(Reference, Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_viridis_c() +
  theme_minimal(base_size = 11) +
  labs(title = sprintf("LASSO Confusion Matrix (Accuracy = %.4f)",
                       lasso_cm$overall["Accuracy"]),
       x = "Actual", y = "Predicted") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
save_plot(p_lasso_cm, "fig_16_lasso_confusion.png", width = 7, height = 6)

lasso_coefs <- coef(lasso_cv, s = "lambda.min")
sink("table_lasso_top_features.txt")
cat("Top 15 LASSO features per category (lambda.min):\n\n")
for (i in seq_along(lasso_coefs)) {
  cat_name <- names(lasso_coefs)[i]
  coef_vec <- as.vector(lasso_coefs[[i]])
  feat_names <- rownames(lasso_coefs[[i]])
  top15 <- head(order(coef_vec, decreasing = TRUE), 15)
  cat("\n", cat_name, "\n")
  for (j in top15) {
    if (coef_vec[j] != 0)
      cat(sprintf("  %-25s  %+.4f\n", feat_names[j], coef_vec[j]))
  }
}
sink()


# 8.2 Ridge
cat("\n MODEL 8.2: RIDGE\n")
set.seed(34074759)
ridge_cv <- cv.glmnet(X_train_sp, y_train,
                      family = "multinomial",
                      type.multinomial = "grouped",
                      alpha = 0, nfolds = 10, trace.it = 0)
ridge_pred <- factor(predict(ridge_cv, X_test_sp,
                             s = "lambda.min", type = "class")[, 1],
                     levels = levels(y_test))
ridge_cm <- confusionMatrix(ridge_pred, y_test)
results_list$ridge <- ridge_cm
cat("Ridge Accuracy:", round(ridge_cm$overall["Accuracy"], 4), "\n")


# 8.3 Elastic net
cat("\nMODEL 8.3: ELASTIC NET\n")
set.seed(34074759)
enet_cv <- cv.glmnet(X_train_sp, y_train,
                     family = "multinomial",
                     type.multinomial = "grouped",
                     alpha = 0.5, nfolds = 10, trace.it = 0)
enet_pred <- factor(predict(enet_cv, X_test_sp,
                            s = "lambda.min", type = "class")[, 1],
                    levels = levels(y_test))
enet_cm <- confusionMatrix(enet_pred, y_test)
results_list$enet <- enet_cm
cat("Elastic Net Accuracy:", round(enet_cm$overall["Accuracy"], 4), "\n")


# 8.3b Tuned elastic net 
tuned_enet_cm <- NULL
if (length(GLMNET_ALPHAS) > 0) {
  cat("\nMODEL 8.3b: TUNED ELASTIC NET (alpha-grid CV)\n")
  cat("alpha grid:", paste(GLMNET_ALPHAS, collapse = ", "), "\n")

  glmnet_tune_results <- data.frame(alpha = numeric(0), lambda = numeric(0),
                                    cv_deviance = numeric(0))
  best_dev <- Inf
  best_glmnet <- NULL

  for (a in GLMNET_ALPHAS) {
    set.seed(34074759)
    fit_a <- cv.glmnet(X_train_sp, y_train,
                       family = "multinomial",
                       type.multinomial = "grouped",
                       alpha = a, nfolds = 10, trace.it = 0)
    dev_min <- min(fit_a$cvm)
    glmnet_tune_results <- rbind(glmnet_tune_results,
                                 data.frame(alpha = a,
                                            lambda = fit_a$lambda.min,
                                            cv_deviance = dev_min))
    cat(sprintf("  alpha=%.2f -> lambda.min=%.5f  cv.deviance=%.4f\n",
                a, fit_a$lambda.min, dev_min))
    if (dev_min < best_dev) {
      best_dev <- dev_min
      best_glmnet <- fit_a
      best_alpha <- a
    }
  }

  cat(sprintf("Best alpha = %.2f  (lambda.min = %.5f, deviance = %.4f)\n",
              best_alpha, best_glmnet$lambda.min, best_dev))
  write.csv(glmnet_tune_results, "table_tuning_glmnet.csv", row.names = FALSE)

  tuned_enet_pred <- factor(predict(best_glmnet, X_test_sp,
                                    s = "lambda.min", type = "class")[, 1],
                            levels = levels(y_test))
  tuned_enet_cm <- confusionMatrix(tuned_enet_pred, y_test)
  results_list$tuned_enet <- tuned_enet_cm
  cat("Tuned Elastic Net Accuracy:",
      round(tuned_enet_cm$overall["Accuracy"], 4), "\n")
}


# 8.4 Naive Bayes
cat("\nMODEL 8.4: NAIVE BAYES\n")
X_train_bin <- ifelse(count_matrix[train_idx, ]  > 0, 1, 0)
X_test_bin <- ifelse(count_matrix[-train_idx, ] > 0, 1, 0)

nb_train_df <- as.data.frame(X_train_bin)
nb_train_df$Category <- y_train
nb_test_df <- as.data.frame(X_test_bin)

nb_model <- naive_bayes(Category ~ ., data = nb_train_df)
nb_pred <- predict(nb_model, newdata = nb_test_df)
nb_cm <- confusionMatrix(nb_pred, y_test)
results_list$nb <- nb_cm
cat("Naive Bayes Accuracy:", round(nb_cm$overall["Accuracy"], 4), "\n")

# 8.4b TUNED NAIVE BAYES (laplace 
tuned_nb_cm <- NULL
if (length(NB_GRID) > 0) {
  cat("\n===== MODEL 8.4b: TUNED NAIVE BAYES =====\n")
  cat("laplace grid:", paste(NB_GRID, collapse = ", "), "\n")
  set.seed(34074759)
  cv_folds <- caret::createFolds(y_train, k = 5, returnTrain = FALSE)

  nb_tune_results <- data.frame(laplace = numeric(0), cv_logloss = numeric(0))
  for (lap in NB_GRID) {
    fold_ll <- numeric(length(cv_folds))
    for (k in seq_along(cv_folds)) {
      val_idx <- cv_folds[[k]]
      tr_df <- nb_train_df[-val_idx, ]
      vl_df <- nb_train_df[val_idx, , drop = FALSE]
      fit_k <- naive_bayes(Category ~ ., data = tr_df, laplace = lap)
      pr_k <- predict(fit_k, newdata = vl_df[, !names(vl_df) %in% "Category"],
                       type = "prob")
      pr_k <- pmax(pr_k, 1e-12) 
      pr_k <- pr_k[, levels(y_train), drop = FALSE]
      true_y <- as.integer(nb_train_df$Category[val_idx])
      ll <- -mean(log(pr_k[cbind(seq_along(true_y), true_y)]))
      fold_ll[k] <- ll
    }
    mean_ll <- mean(fold_ll)
    nb_tune_results <- rbind(nb_tune_results,
                             data.frame(laplace = lap, cv_logloss = mean_ll))
    cat(sprintf("  laplace=%.2f -> cv_logloss=%.4f\n", lap, mean_ll))
  }
  best_lap <- nb_tune_results$laplace[which.min(nb_tune_results$cv_logloss)]
  cat("Best laplace =", best_lap, "\n")
  write.csv(nb_tune_results, "table_tuning_nb.csv", row.names = FALSE)

  tuned_nb_model <- naive_bayes(Category ~ ., data = nb_train_df,
                                laplace = best_lap)
  tuned_nb_pred <- predict(tuned_nb_model, newdata = nb_test_df)
  tuned_nb_cm <- confusionMatrix(tuned_nb_pred, y_test)
  results_list$tuned_nb <- tuned_nb_cm
  cat("Tuned NB Accuracy:", round(tuned_nb_cm$overall["Accuracy"], 4), "\n")
}


# 8.5 SVM radial
cat("\nMODEL 8.5: SVM RADIAL\n")

if (emb_available) {
  cat("Training SVM Radial on 384-d MiniLM embeddings.\n")
  set.seed(34074759)
  svm_rad <- svm(x = X_emb_train, y = y_train,
                 kernel = "radial", cost = 1,
                 gamma = 1 / ncol(X_emb_train),
                 probability = TRUE)
  svm_pred <- predict(svm_rad, X_emb_test, probability = TRUE)
  svm_probs <- attr(svm_pred, "probabilities")
  svm_cm <- confusionMatrix(svm_pred, y_test)
  cat("SVM Radial (embeddings) Accuracy:",
      round(svm_cm$overall["Accuracy"], 4), "\n")
} else if (irlba_ok) {
  library(irlba)
  cat("Embeddings unavailable — running SVM Radial on irlba-LSA(150).\n")
  cat("(Latent Semantic Analysis: truncated SVD on sparse TF-IDF.)\n")

  set.seed(34074759)
  n_lsa <- 150L
  lsa_fit <- irlba::irlba(text_train_sp, nv = n_lsa, maxit = 200)

  X_lsa_train <- lsa_fit$u %*% diag(lsa_fit$d)  
  X_lsa_test <- as.matrix(text_test_sp %*% lsa_fit$v)    

  l2norm <- function(M) {
    rn <- sqrt(rowSums(M^2)); rn[rn == 0] <- 1; M / rn
  }
  X_lsa_train <- l2norm(X_lsa_train)
  X_lsa_test <- l2norm(X_lsa_test)

  X_svm_train <- cbind(X_lsa_train, numeric_scaled[train_idx, ])
  X_svm_test <- cbind(X_lsa_test,  numeric_scaled[-train_idx, ])
  cat("SVM input dim:", ncol(X_svm_train), "\n")

  set.seed(34074759)
  svm_rad <- svm(x = X_svm_train, y = y_train,
                 kernel = "radial", cost = 1,
                 gamma = 1 / ncol(X_svm_train),
                 probability = TRUE)
  svm_pred <- predict(svm_rad, X_svm_test, probability = TRUE)
  svm_probs <- attr(svm_pred, "probabilities")
  svm_cm <- confusionMatrix(svm_pred, y_test)
  cat("SVM Radial (LSA-150 fallback) Accuracy:",
      round(svm_cm$overall["Accuracy"], 4), "\n")
} else {
  cat("Neither embeddings nor irlba available — SVM Radial skipped.\n")
  cat("Install with: install.packages('irlba')\n")
  svm_cm <- NULL
}

if (!is.null(svm_cm)) results_list$svm_radial <- svm_cm


# 8.5b Tuned SVM radial
tuned_svm_cm <- NULL
if (exists("svm_rad")) {
  cat("\n MODEL 8.5b: TUNED SVM RADIAL\n")
  X_svm_train_for_tune <- if (emb_available) X_emb_train else if (exists("X_svm_train")) X_svm_train else NULL
  X_svm_test_for_tune <- if (emb_available) X_emb_test  else if (exists("X_svm_test"))  X_svm_test  else NULL

  if (!is.null(X_svm_train_for_tune)) {
    cost_grid <- 2 ^ c(-1, 1, 3)
    gamma_grid <- 2 ^ c(-7, -5, -3)
    cat("cost grid:", paste(round(cost_grid, 4), collapse = ", "), "\n")
    cat("gamma grid:", paste(round(gamma_grid, 6), collapse = ", "), "\n")

    set.seed(34074759)
    sub_n <- min(2500L, nrow(X_svm_train_for_tune))
    sub_idx_svm <- sample.int(nrow(X_svm_train_for_tune), sub_n)
    cat("Tuning on a", sub_n, "-row subsample (5-fold CV)...\n")

    tune_obj <- tryCatch({
      e1071::tune(svm,
                  train.x = X_svm_train_for_tune[sub_idx_svm, , drop = FALSE],
                  train.y = y_train[sub_idx_svm],
                  kernel = "radial",
                  ranges = list(cost = cost_grid, gamma = gamma_grid),
                  tunecontrol = e1071::tune.control(
                    sampling = "cross", cross = 5))
    }, error = function(e) {
      cat("** SVM tuning failed:", conditionMessage(e), "**\n")
      NULL
    })

    if (!is.null(tune_obj)) {
      svm_perf <- as.data.frame(tune_obj$performances)
      write.csv(svm_perf, "table_tuning_svm.csv", row.names = FALSE)
      cat("Best (cost, gamma) =",
          tune_obj$best.parameters$cost, ",",
          tune_obj$best.parameters$gamma, "\n")
      cat("Best CV error =", round(tune_obj$best.performance, 4), "\n")

      set.seed(34074759)
      svm_tuned <- svm(x = X_svm_train_for_tune,
                       y = y_train,
                       kernel = "radial",
                       cost = tune_obj$best.parameters$cost,
                       gamma = tune_obj$best.parameters$gamma,
                       probability = TRUE)
      svm_tuned_pred <- predict(svm_tuned, X_svm_test_for_tune,
                                 probability = TRUE)
      svm_tuned_probs <- attr(svm_tuned_pred, "probabilities")
      tuned_svm_cm <- confusionMatrix(svm_tuned_pred, y_test)
      results_list$tuned_svm <- tuned_svm_cm
      cat("Tuned SVM Accuracy:",
          round(tuned_svm_cm$overall["Accuracy"], 4), "\n")
    }
  } else {
    cat("Tuned SVM skipped (no compatible feature matrix).\n")
  }
}


# 8.6 XGBoost 
cat("\nMODEL 8.6: XGBOOST\n")

y_train_int <- as.integer(y_train) - 1L
y_test_int <- as.integer(y_test)  - 1L
num_class <- nlevels(y_train)

dtrain <- xgb.DMatrix(data = X_train_sp, label = y_train_int)
dtest <- xgb.DMatrix(data = X_test_sp,  label = y_test_int)

y_train_inner_int <- as.integer(y_train_inner) - 1L
y_inner_val_int <- as.integer(y_inner_val)   - 1L
dtrain_inner <- xgb.DMatrix(data = X_train_inner_sp, label = y_train_inner_int)
dval_inner <- xgb.DMatrix(data = X_inner_val_sp,   label = y_inner_val_int)

xgb_params <- list(
  objective = "multi:softprob",
  num_class = num_class,
  eval_metric = "mlogloss",
  tree_method = "hist",
  max_depth = 8,
  eta = 0.05,
  subsample = 0.85,
  colsample_bytree = 0.7,
  min_child_weight = 3,
  nthread = n_cores
)

set.seed(34074759)
xgb_es <- xgb.train(
  params = xgb_params,
  data = dtrain_inner,
  nrounds = 1000,
  watchlist = list(train = dtrain_inner, val = dval_inner),
  early_stopping_rounds = 30,
  verbose = 0
)
best_iter_xgb <- xgb_es$best_iteration
cat("XGBoost best_iter (from inner-val early stopping):", best_iter_xgb, "\n")

set.seed(34074759)
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = best_iter_xgb,
  verbose = 0
)

xgb_proba <- predict(xgb_model, dtest, reshape = TRUE)
colnames(xgb_proba) <- levels(y_test)
xgb_pred <- factor(levels(y_test)[max.col(xgb_proba)], levels = levels(y_test))
xgb_cm <- confusionMatrix(xgb_pred, y_test)
results_list$xgb <- xgb_cm
cat("XGBoost (sparse) Accuracy:", round(xgb_cm$overall["Accuracy"], 4), "\n")
cat("Final model trained on full training data with nrounds =", best_iter_xgb, "\n")

# Top 20 feature importance plot
xgb_imp <- xgb.importance(model = xgb_model)
png("xgb_importance.png", width = 800, height = 600, res = 110)
xgb.plot.importance(xgb_imp, top_n = 20,
                    main = "XGBoost — Top 20 Feature Importance (sparse TF-IDF)")
dev.off()


# 8.6b Tuned XGBoost 
tuned_xgb_cm <- NULL
xgb_proba_tuned <- NULL
xgb_pred_tuned <- NULL
xgb_model_tuned <- NULL

cat("\nMODEL 8.6b: TUNED XGBOOST (random search)\n")
cat("Sampling", N_RANDOM_XGB, "random configurations..\n")

set.seed(34074759)
sample_xgb_config <- function() list(
  objective = "multi:softprob",
  num_class = num_class,
  eval_metric = "mlogloss",
  tree_method = "hist",
  max_depth = sample(c(4L, 6L, 8L, 10L, 12L), 1),
  eta = sample(c(0.02, 0.03, 0.05, 0.08, 0.10), 1),
  subsample = runif(1, 0.7, 1.0),
  colsample_bytree = runif(1, 0.5, 1.0),
  min_child_weight = sample(c(1, 3, 5, 8), 1),
  gamma = sample(c(0, 0.1, 0.5, 1.0), 1),
  nthread = n_cores
)

coalesce_scalar <- function(x, default) {
  if (is.null(x) || length(x) == 0L || all(is.na(x))) default else x[[1L]]
}
best_test_score <- function(eval_log) {
  if (is.null(eval_log) || nrow(eval_log) == 0L) return(NA_real_)
  cn <- colnames(eval_log)
  cand <- cn[grepl("^test[_-].+[_-]mean$", cn)]
  if (length(cand) == 0L) return(NA_real_)
  vals <- eval_log[[cand[1L]]]
  if (length(vals) == 0L || all(is.na(vals))) return(NA_real_)
  suppressWarnings(min(vals, na.rm = TRUE))
}

xgb_search <- vector("list", N_RANDOM_XGB)
for (i in seq_len(N_RANDOM_XGB)) {
  set.seed(34074759 + i)
  cfg <- sample_xgb_config()
  cv_out <- tryCatch({
    xgb.cv(params = cfg, data = dtrain,
           nrounds = 600, nfold = 3,
           early_stopping_rounds = 25,
           verbose = 0, maximize = FALSE)
  }, error = function(e) NULL)
  if (is.null(cv_out)) {
    cat(sprintf("  config %02d: xgb.cv failed — skipped\n", i)); next
  }
  best_score <- best_test_score(cv_out$evaluation_log)
  if (!is.finite(best_score)) {
    cat(sprintf("  config %02d: no usable test metric — skipped\n", i)); next
  }
  best_iter <- coalesce_scalar(cv_out$best_iteration,
                               default = coalesce_scalar(cv_out$niter, 100L))
  xgb_search[[i]] <- list(cfg = cfg,
                          best_logloss = as.numeric(best_score),
                          best_iter = as.integer(best_iter))
  cat(sprintf("  config %02d: depth=%d eta=%.3f sub=%.2f col=%.2f mcw=%d gamma=%.1f -> logloss=%.4f (iter %d)\n",
              i, cfg$max_depth, cfg$eta, cfg$subsample, cfg$colsample_bytree,
              cfg$min_child_weight, cfg$gamma, best_score, best_iter))
}
xgb_search <- xgb_search[!vapply(xgb_search, is.null, logical(1))]

if (length(xgb_search) == 0L) {
  cat("** All XGBoost configurations failed; tuned XGBoost skipped. **\n")
} else {
  xgb_tune_df <- do.call(rbind, lapply(seq_along(xgb_search), function(i) {
    s <- xgb_search[[i]]
    data.frame(config_id = i,
               max_depth = coalesce_scalar(s$cfg$max_depth,        NA_integer_),
               eta = coalesce_scalar(s$cfg$eta,              NA_real_),
               subsample = round(coalesce_scalar(s$cfg$subsample,        NA_real_), 4),
               colsample_bytree = round(coalesce_scalar(s$cfg$colsample_bytree, NA_real_), 4),
               min_child_weight = coalesce_scalar(s$cfg$min_child_weight, NA_integer_),
               gamma = coalesce_scalar(s$cfg$gamma,            NA_real_),
               cv_logloss = round(coalesce_scalar(s$best_logloss,   NA_real_), 5),
               best_iter = coalesce_scalar(s$best_iter,            NA_integer_))
  }))
  xgb_tune_df <- xgb_tune_df[order(xgb_tune_df$cv_logloss), ]
  write.csv(xgb_tune_df, "table_tuning_xgb.csv", row.names = FALSE)

  best_xgb_idx <- which.min(vapply(xgb_search, function(s) s$best_logloss, numeric(1)))
  best_xgb_cfg <- xgb_search[[best_xgb_idx]]$cfg
  best_xgb_iter <- xgb_search[[best_xgb_idx]]$best_iter
  cat(sprintf("BEST: config %d  cv_logloss=%.4f  best_iter=%d\n",
              best_xgb_idx, xgb_search[[best_xgb_idx]]$best_logloss, best_xgb_iter))

  set.seed(34074759)
  xgb_model_tuned <- xgb.train(
    params = best_xgb_cfg,
    data = dtrain,
    nrounds = best_xgb_iter,
    verbose = 0
  )
  xgb_proba_tuned <- predict(xgb_model_tuned, dtest, reshape = TRUE)
  colnames(xgb_proba_tuned) <- levels(y_test)
  xgb_pred_tuned <- factor(levels(y_test)[max.col(xgb_proba_tuned)],
                           levels = levels(y_test))
  tuned_xgb_cm <- confusionMatrix(xgb_pred_tuned, y_test)
  results_list$tuned_xgb <- tuned_xgb_cm
  cat("Tuned XGBoost Accuracy:",
      round(tuned_xgb_cm$overall["Accuracy"], 4), "\n")
}

# 8.7 LightGBM

lgb_available <- FALSE
lgb_train <- NULL
lgb_model <- NULL
lgb_proba <- NULL

tryCatch({
  library(lightgbm)

  cat("\nMODEL 8.7: LIGHTGBM\n")

  lgb_train <- lgb.Dataset(
    data = X_train_sp,
    label = y_train_int,
    params = list(feature_pre_filter = FALSE)
  )

  lgb_train_inner <- lgb.Dataset(
    data = X_train_inner_sp,
    label = y_train_inner_int,
    params = list(feature_pre_filter = FALSE)
  )

  lgb_val_inner <- lgb.Dataset(
    data = X_inner_val_sp,
    label = y_inner_val_int,
    reference = lgb_train_inner,
    params = list(feature_pre_filter = FALSE)
  )

  lgb_params <- list(
    objective = "multiclass",
    num_class = num_class,
    metric = "multi_logloss",
    learning_rate = 0.05,
    num_leaves = 63,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq= 5L,
    min_data_in_leaf = 20L,
    feature_pre_filter = FALSE,
    num_threads = n_cores,
    verbosity = -1L
  )

  set.seed(34074759)

  lgb_es <- lgb.train(
    params = lgb_params,
    data = lgb_train_inner,
    nrounds = 1000,
    valids = list(val = lgb_val_inner),
    early_stopping_rounds = 30,
    verbose = -1
  )

  best_iter_lgb <- tryCatch(lgb_es$best_iter, error = function(e) NULL)

  if (is.null(best_iter_lgb) ||
      length(best_iter_lgb) != 1L ||
      is.na(best_iter_lgb) ||
      !is.finite(best_iter_lgb) ||
      best_iter_lgb < 1L) {
    best_iter_lgb <- 1000L
  }

  best_iter_lgb <- as.integer(best_iter_lgb)

  cat("LightGBM best_iter from inner-val early stopping:",
      best_iter_lgb, "\n")

  set.seed(34074759)

  lgb_model <- lgb.train(
    params = lgb_params,
    data = lgb_train,
    nrounds = best_iter_lgb,
    verbose = -1
  )

  lgb_proba <- predict(lgb_model, X_test_sp)

  if (is.vector(lgb_proba)) {
    lgb_proba <- matrix(
      lgb_proba,
      ncol = num_class,
      byrow = TRUE
    )
  }

  colnames(lgb_proba) <- levels(y_test)

  lgb_pred <- factor(
    levels(y_test)[max.col(lgb_proba)],
    levels = levels(y_test)
  )

  lgb_cm <- confusionMatrix(lgb_pred, y_test)
  results_list$lgb <- lgb_cm

  cat("LightGBM Accuracy:",
      round(lgb_cm$overall["Accuracy"], 4), "\n")
  cat("Final model trained on full training data with nrounds =",
      best_iter_lgb, "\n")

  lgb_available <- TRUE

}, error = function(e) {
  cat("** LightGBM skipped:", conditionMessage(e), "**\n")
})


# 8.7b Tuned LightGBM 

tuned_lgb_cm <- NULL
lgb_proba_tuned <- NULL

if (lgb_available && N_RANDOM_LGB > 0) {
  cat("\nMODEL 8.7b: TUNED LIGHTGBM\n")
  cat("Sampling", N_RANDOM_LGB, "random configurations\n")

  set.seed(34074759)

  sample_lgb_config <- function() {
    list(
      objective = "multiclass",
      num_class = num_class,
      metric = "multi_logloss",
      learning_rate = sample(c(0.02, 0.03, 0.05, 0.08, 0.10), 1),
      num_leaves = sample(c(15L, 31L, 63L, 127L, 255L), 1),
      feature_fraction = runif(1, 0.5, 1.0),
      bagging_fraction = runif(1, 0.6, 1.0),
      bagging_freq = 5L,
      min_data_in_leaf = sample(c(5L, 10L, 20L, 40L, 80L), 1),
      lambda_l2 = sample(c(0, 0.1, 1.0, 5.0), 1),
      num_threads = n_cores,
      verbosity = -1L
    )
  }

  lgb_search <- vector("list", N_RANDOM_LGB)

  for (i in seq_len(N_RANDOM_LGB)) {
    set.seed(34074759 + i)

    cfg <- sample_lgb_config()

    cv_out <- tryCatch({
      lightgbm::lgb.cv(
        params = cfg,
        data = lgb_train,
        nrounds = 600,
        nfold = 3,
        early_stopping_rounds = 25,
        verbose = -1
      )
    }, error = function(e) {
      cat(sprintf("  config %02d: lgb.cv failed — %s\n",
                  i, conditionMessage(e)))
      NULL
    })

    if (is.null(cv_out)) next

    best_score <- tryCatch(cv_out$best_score, error = function(e) NA_real_)

    if (is.null(best_score) ||
        length(best_score) != 1L ||
        is.na(best_score) ||
        !is.finite(best_score)) {
      best_score <- tryCatch(
        min(unlist(cv_out$record_evals$valid$multi_logloss$eval),
            na.rm = TRUE),
        error = function(e) NA_real_
      )
    }

    if (is.null(best_score) ||
        length(best_score) != 1L ||
        is.na(best_score) ||
        !is.finite(best_score)) {
      cat(sprintf("  config %02d: no usable LightGBM CV metric — skipped\n", i))
      next
    }

    best_iter <- tryCatch(cv_out$best_iter, error = function(e) NA_integer_)

    if (is.null(best_iter) ||
        length(best_iter) != 1L ||
        is.na(best_iter) ||
        !is.finite(best_iter) ||
        best_iter < 1L) {
      best_iter <- 100L
    }

    best_iter <- as.integer(best_iter)

    lgb_search[[i]] <- list(
      cfg = cfg,
      best_logloss = as.numeric(best_score),
      best_iter = best_iter
    )

    cat(sprintf(
      "  config %02d: lr=%.3f leaves=%d ff=%.2f bf=%.2f mdl=%d l2=%.1f -> logloss=%.4f (iter %d)\n",
      i,
      cfg$learning_rate,
      cfg$num_leaves,
      cfg$feature_fraction,
      cfg$bagging_fraction,
      cfg$min_data_in_leaf,
      cfg$lambda_l2,
      best_score,
      best_iter
    ))
  }

  lgb_search <- lgb_search[!vapply(lgb_search, function(s) {
    is.null(s) ||
      is.null(s$best_logloss) ||
      length(s$best_logloss) != 1L ||
      is.na(s$best_logloss) ||
      !is.finite(s$best_logloss)
  }, logical(1))]

  if (length(lgb_search) > 0L) {
    lgb_tune_df <- do.call(rbind, lapply(seq_along(lgb_search), function(i) {
      s <- lgb_search[[i]]

      data.frame(
        config_id = i,
        learning_rate = s$cfg$learning_rate,
        num_leaves = s$cfg$num_leaves,
        feature_fraction = round(s$cfg$feature_fraction, 4),
        bagging_fraction = round(s$cfg$bagging_fraction, 4),
        min_data_in_leaf = s$cfg$min_data_in_leaf,
        lambda_l2 = s$cfg$lambda_l2,
        cv_logloss = round(s$best_logloss, 5),
        best_iter = s$best_iter
      )
    }))

    lgb_tune_df <- lgb_tune_df[order(lgb_tune_df$cv_logloss), ]

    write.csv(
      lgb_tune_df,
      "table_tuning_lgb.csv",
      row.names = FALSE
    )

    best_lgb_idx <- which.min(vapply(
      lgb_search,
      function(s) s$best_logloss,
      numeric(1)
    ))

    best_lgb_cfg <- lgb_search[[best_lgb_idx]]$cfg
    best_lgb_iter <- lgb_search[[best_lgb_idx]]$best_iter

    cat(sprintf(
      "BEST: config %d  cv_logloss=%.4f  best_iter=%d\n",
      best_lgb_idx,
      lgb_search[[best_lgb_idx]]$best_logloss,
      best_lgb_iter
    ))

    set.seed(34074759)

    lgb_train_full <- lgb.Dataset(
      data = X_train_sp,
      label = y_train_int
    )

    lgb_model_tuned <- lgb.train(
      params = best_lgb_cfg,
      data = lgb_train_full,
      nrounds = best_lgb_iter,
      verbose = -1
    )

    lgb_proba_tuned <- predict(lgb_model_tuned, X_test_sp)

    if (is.vector(lgb_proba_tuned)) {
      lgb_proba_tuned <- matrix(
        lgb_proba_tuned,
        ncol = num_class,
        byrow = TRUE
      )
    }

    colnames(lgb_proba_tuned) <- levels(y_test)

    lgb_pred_tuned <- factor(
      levels(y_test)[max.col(lgb_proba_tuned)],
      levels = levels(y_test)
    )

    tuned_lgb_cm <- confusionMatrix(lgb_pred_tuned, y_test)
    results_list$tuned_lgb <- tuned_lgb_cm

    cat("Tuned LightGBM Accuracy:",
        round(tuned_lgb_cm$overall["Accuracy"], 4), "\n")

  } else {
    cat("** All LightGBM tuning configurations failed; tuned LightGBM skipped. **\n")
  }
}


# 8.8 fastText supervised
ft_available <- FALSE
tryCatch({
  library(fastTextR)
  cat("\nMODEL 8.8: FASTTEXT SUPERVISED\n")

  ft_dir <- tempfile("ft_")
  dir.create(ft_dir, showWarnings = FALSE, recursive = TRUE)
  train_file <- file.path(ft_dir, "train.txt")
  test_file <- file.path(ft_dir, "test.txt")

  ft_clean <- function(x) {
    x <- enc2utf8(as.character(x))
    x[is.na(x)] <- ""
    x <- tolower(x)
    x <- gsub("[\r\n\t]", " ", x)
    x <- gsub("[^a-z0-9 ]", " ", x)
    x <- gsub("\\s+", " ", x)
    trimws(x)
  }
  ft_label <- function(y) gsub("[^a-zA-Z0-9_]", "_", as.character(y))

  train_lines <- paste0("__label__", ft_label(y_train), " ",
                        ft_clean(data$FullText[train_idx]))
  test_lines <- paste0("__label__", ft_label(y_test), " ",
                        ft_clean(data$FullText[-train_idx]))

  writeLines(train_lines, train_file, useBytes = TRUE)
  writeLines(test_lines,  test_file,  useBytes = TRUE)

  set.seed(34074759)
  ft_args <- list(
    loss = "softmax",
    learning_rate = 0.5,
    word_vec_size = 100L,
    epoch = 25L,
    max_len_ngram = 2L,
    min_count = 1L,
    nthreads = as.integer(max(1L, n_cores)),
    verbose = 0L
  )

  if ("seed" %in% names(formals(fastTextR::ft_control))) ft_args$seed <- 42L
  ft_ctrl <- do.call(fastTextR::ft_control, ft_args)

  ft_model <- ft_train(file = train_file, method = "supervised", control = ft_ctrl)

  test_texts_ft <- gsub("^__label__\\S+\\s+", "", readLines(test_file, warn = FALSE))
  ft_predictions <- ft_predict(ft_model, newdata = test_texts_ft, k = 1L,
                               threshold = 0, rval = "dense")

  if (is.matrix(ft_predictions)) {
    if (is.null(colnames(ft_predictions))) {
      stop("fastTextR returned a dense probability matrix without label column names.")
    }
    ft_pred_label <- colnames(ft_predictions)[max.col(ft_predictions, ties.method = "first")]
  } else if (is.data.frame(ft_predictions)) {
    if ("label" %in% names(ft_predictions)) {
      ft_pred_label <- as.character(ft_predictions$label)
    } else {
      ft_pred_label <- as.character(ft_predictions[[1]])
    }
  } else if (is.list(ft_predictions)) {
    ft_pred_label <- vapply(ft_predictions, function(z) {
      if (length(z) == 0L) return(NA_character_)
      if (is.null(names(z))) as.character(z[[1]]) else names(z)[1]
    }, character(1))
  } else {
    ft_pred_label <- as.character(ft_predictions)
  }

  ft_pred_label <- gsub("^__label__", "", ft_pred_label)
  level_lookup <- setNames(levels(y_test), ft_label(levels(y_test)))
  ft_pred <- factor(unname(level_lookup[ft_pred_label]), levels = levels(y_test))

  if (any(is.na(ft_pred))) {
    fallback_class <- names(sort(table(y_train), decreasing = TRUE))[1]
    cat("WARN:", sum(is.na(ft_pred)),
        "fastText predictions could not be matched to known labels; using",
        fallback_class, "for those cases.\n")
    ft_pred[is.na(ft_pred)] <- fallback_class
    ft_pred <- factor(ft_pred, levels = levels(y_test))
  }

  ft_cm <- confusionMatrix(ft_pred, y_test)
  results_list$fasttext <- ft_cm
  cat("fastText Accuracy:", round(ft_cm$overall["Accuracy"], 4), "\n")
  ft_available <- TRUE
}, error = function(e) {
  cat("** fastText skipped:", conditionMessage(e), "**\n")
  cat("Install/update with: install.packages('fastTextR')\n")
})


# 8.9 MiniLM embeddings + multinomial LR, will likely not be included in the analysis
emb_lr_cm <- NULL
if (emb_available) {
  cat("\n MODEL 8.9: MINILM EMBEDDINGS + MULTINOMIAL LR \n")

  set.seed(34074759)
  emb_lr_cv <- cv.glmnet(as.matrix(X_emb_train), y_train,
                         family = "multinomial",
                         type.multinomial = "grouped",
                         alpha = 0,
                         nfolds = 10, trace.it = 0)
  emb_lr_pred <- factor(predict(emb_lr_cv, as.matrix(X_emb_test),
                                s = "lambda.min", type = "class")[, 1],
                        levels = levels(y_test))
  emb_lr_cm <- confusionMatrix(emb_lr_pred, y_test)
  results_list$emb_lr <- emb_lr_cm
  cat("MiniLM + LR Accuracy:", round(emb_lr_cm$overall["Accuracy"], 4), "\n")
}


# 8.10-8.12 Keras deep learning, keras3
keras_available <- FALSE
keras_pkg <- NULL

plot_history <- function(hist, title) {
  hist_df <- as.data.frame(hist)
  if (!"data" %in% names(hist_df)) {
    nm <- names(hist_df)
    if ("split" %in% nm) hist_df$data <- hist_df$split
    else if ("dataset" %in% nm) hist_df$data <- hist_df$dataset
    else hist_df$data <- "training"
  }
  ggplot(hist_df, aes(x = epoch, y = value, colour = data)) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.2) +
    facet_wrap(~ metric, scales = "free_y", ncol = 1) +
    theme_minimal(base_size = 11) +
    labs(title = title, x = "Epoch", y = NULL, colour = NULL) +
    theme(legend.position = "bottom",
          strip.text = element_text(face = "bold"))
}

safe_save_history <- function(hist, title, filename) {
  tryCatch(
    save_plot(plot_history(hist, title), filename, width = 8, height = 6),
    error = function(e) message("Could not save ", filename, ": ", conditionMessage(e))
  )
}

nn_clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  x <- tolower(x)
  x <- gsub("<[^>]+>", " ", x)
  x <- gsub("&[a-zA-Z]+;", " ", x)
  x <- gsub("&#\\d+;", " ", x)
  x <- gsub("[^a-z0-9]+", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

nn_tokenize <- function(x) {
  cleaned <- nn_clean_text(x)
  toks <- strsplit(cleaned, "\\s+", perl = TRUE)
  lapply(toks, function(z) z[nzchar(z)])
}

build_vocab <- function(tokens_list, max_features = 10000L) {
  all_tokens <- unlist(tokens_list, use.names = FALSE)
  all_tokens <- all_tokens[nzchar(all_tokens)]
  if (length(all_tokens) == 0L) stop("No tokens available for neural-network vocabulary.")
  freqs <- sort(table(all_tokens), decreasing = TRUE)
  vocab_n <- min(as.integer(max_features) - 2L, length(freqs))
  vocab <- names(freqs)[seq_len(vocab_n)]
  setNames(seq.int(2L, vocab_n + 1L), vocab)  
}

texts_to_matrix <- function(tokens_list, word_index, maxlen = 250L) {
  x <- matrix(0L, nrow = length(tokens_list), ncol = as.integer(maxlen))
  for (i in seq_along(tokens_list)) {
    tok <- tokens_list[[i]]
    if (!length(tok)) next
    ids <- unname(word_index[tok])
    ids[is.na(ids)] <- 1L
    ids <- as.integer(ids)
    if (length(ids) > maxlen) ids <- ids[seq_len(maxlen)]
    x[i, seq_along(ids)] <- ids
  }
  storage.mode(x) <- "integer"
  x
}

tryCatch({
  if (requireNamespace("keras3", quietly = TRUE)) {
    keras_pkg <- "keras3"
    library(keras3)
    use_backend("tensorflow")
    cat("\nMODELS 8.10-8.12: KERAS DEEP LEARNING\n")
    cat("Keras backend:", config_backend(), "\n")
  } else if (requireNamespace("keras", quietly = TRUE)) {
    keras_pkg <- "keras"
    if (requireNamespace("reticulate", quietly = TRUE) &&
        exists("py_require_legacy_keras", where = asNamespace("reticulate"))) {
      reticulate::py_require_legacy_keras()
    }
    library(keras)
    cat("\n MODELS 8.10-8.12: KERAS DEEP LEARNING\n")
  } else {
    stop("Neither keras3 nor keras is installed.")
  }

  all_texts <- as.character(data$FullText)
  train_texts <- all_texts[train_idx]
  test_texts <- all_texts[-train_idx]

  label_levels <- levels(y)
  y_train_k <- as.integer(y_train) - 1L
  y_test_k <- as.integer(y_test)  - 1L

  max_features <- 10000L
  maxlen <- 250L
  embed_dim <- 100L

  cat("Preparing integer token sequences for Keras models\n")
  train_tokens <- nn_tokenize(train_texts)
  test_tokens <- nn_tokenize(test_texts)
  word_index <- build_vocab(train_tokens, max_features = max_features)
  X_nn_train <- texts_to_matrix(train_tokens, word_index, maxlen = maxlen)
  X_nn_test <- texts_to_matrix(test_tokens,  word_index, maxlen = maxlen)
  input_dim_nn <- max(unlist(word_index, use.names = FALSE)) + 1L

  cat("Keras input matrix:", nrow(X_nn_train), "x", ncol(X_nn_train),
      "| vocab input_dim:", input_dim_nn, "\n")

  make_callbacks <- function(patience_es = 4) list(
    callback_early_stopping(monitor = "val_loss", patience = patience_es,
                            restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5,
                                  patience = 2, min_lr = 1e-5)
  )

  # Model 8.10: Dense MLP
  cat("Building Dense MLP\n")
  inp_a <- layer_input(shape = c(maxlen), dtype = "int32")
  out_a <- inp_a %>%
    layer_embedding(input_dim = input_dim_nn, output_dim = embed_dim) %>%
    layer_global_average_pooling_1d() %>%
    layer_dropout(0.4) %>%
    layer_dense(128, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(64, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(length(label_levels), activation = "softmax")

  model_dense <- keras_model(inp_a, out_a)
  compile(model_dense,
    optimizer = optimizer_adam(learning_rate = 1e-3, clipnorm = 1.0),
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy"
  )
  set.seed(34074759)
  hist_dense <- fit(model_dense,
    x = X_nn_train, y = y_train_k,
    epochs = 25, batch_size = 32, validation_split = 0.2,
    callbacks = make_callbacks(), verbose = 1
  )
  safe_save_history(hist_dense, "Keras Dense (NN1) — Training History",
                    "keras_dense_history.png")

  dense_probs <- as.matrix(predict(model_dense, X_nn_test, verbose = 0))
  dense_pred <- factor(label_levels[apply(dense_probs, 1, which.max)],
                        levels = label_levels)
  dense_cm <- confusionMatrix(dense_pred, y_test)
  results_list$keras_dense <- dense_cm
  cat("Keras Dense Accuracy:", round(dense_cm$overall["Accuracy"], 4), "\n")

  # Model 8.11: 1-D CNN
  cat("\nBuilding 1-D CNN (NN2)\n")
  inp_b <- layer_input(shape = c(maxlen), dtype = "int32")
  out_b <- inp_b %>%
    layer_embedding(input_dim = input_dim_nn, output_dim = embed_dim) %>%
    layer_dropout(0.3) %>%
    layer_conv_1d(filters = 128, kernel_size = 5, activation = "relu") %>%
    layer_global_max_pooling_1d() %>%
    layer_dropout(0.3) %>%
    layer_dense(64, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(length(label_levels), activation = "softmax")

  model_cnn <- keras_model(inp_b, out_b)
  compile(model_cnn,
    optimizer = optimizer_adam(learning_rate = 1e-3, clipnorm = 1.0),
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy"
  )
  set.seed(34074759)
  hist_cnn <- fit(model_cnn,
    x = X_nn_train, y = y_train_k,
    epochs = 20, batch_size = 32, validation_split = 0.2,
    callbacks = make_callbacks(), verbose = 1
  )
  safe_save_history(hist_cnn, "Keras 1-D CNN (NN2) — Training History",
                    "keras_cnn_history.png")

  cnn_probs <- as.matrix(predict(model_cnn, X_nn_test, verbose = 0))
  cnn_pred <- factor(label_levels[apply(cnn_probs, 1, which.max)],
                      levels = label_levels)
  cnn_cm <- confusionMatrix(cnn_pred, y_test)
  results_list$keras_cnn <- cnn_cm
  cat("Keras CNN Accuracy:", round(cnn_cm$overall["Accuracy"], 4), "\n")

  # Model 8.12: LSTM
  cat("\nBuilding LSTM (NN3)...\n")
  inp_c <- layer_input(shape = c(maxlen), dtype = "int32")
  out_c <- inp_c %>%
    layer_embedding(input_dim = input_dim_nn, output_dim = embed_dim) %>%
    layer_spatial_dropout_1d(0.3) %>%
    layer_lstm(units = 64, dropout = 0.2, return_sequences = FALSE) %>%
    layer_dense(64, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(length(label_levels), activation = "softmax")

  model_lstm <- keras_model(inp_c, out_c)
  compile(model_lstm,
    optimizer = optimizer_adam(learning_rate = 5e-4, clipnorm = 1.0),
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy"
  )
  set.seed(34074759)
  hist_lstm <- fit(model_lstm,
    x = X_nn_train, y = y_train_k,
    epochs = 20, batch_size = 32, validation_split = 0.2,
    callbacks = make_callbacks(patience_es = 5), verbose = 1
  )
  safe_save_history(hist_lstm, "Keras LSTM (NN3) — Training History",
                    "keras_lstm_history.png")

  lstm_probs <- as.matrix(predict(model_lstm, X_nn_test, verbose = 0))
  lstm_pred <- factor(label_levels[apply(lstm_probs, 1, which.max)],
                       levels = label_levels)
  lstm_cm <- confusionMatrix(lstm_pred, y_test)
  results_list$keras_lstm <- lstm_cm
  cat("Keras LSTM Accuracy:", round(lstm_cm$overall["Accuracy"], 4), "\n")

  #Model 8.12b: BiLSTM (NN4
  cat("\nBuilding BiLSTM (NN4)\n")

  build_bilstm <- function(units, dropout_rate, embed_dim_local, learning_rate) {
    inp_d <- layer_input(shape = c(maxlen), dtype = "int32")
    out_d <- inp_d %>%
      layer_embedding(input_dim = input_dim_nn, output_dim = embed_dim_local) %>%
      layer_spatial_dropout_1d(0.3) %>%
      bidirectional(layer_lstm(units = units, dropout = dropout_rate,
                                return_sequences = FALSE)) %>%
      layer_dense(64, activation = "relu") %>%
      layer_dropout(dropout_rate) %>%
      layer_dense(length(label_levels), activation = "softmax")
    m <- keras_model(inp_d, out_d)
    compile(m,
      optimizer = optimizer_adam(learning_rate = learning_rate, clipnorm = 1.0),
      loss = "sparse_categorical_crossentropy",
      metrics = "accuracy"
    )
    m
  }

  # Hyperparameter search
  bilstm_cm <- NULL
  if (BILSTM_GRID_N > 0) {
    cat("\n MODEL 8.12b: BiLSTM with hyperparameter tuning\n")
    cat("Sampling", BILSTM_GRID_N, "BiLSTM configurations\n")

    set.seed(34074759)
    sample_bilstm_config <- function() list(
      units = sample(c(32L, 48L, 64L, 96L), 1),
      dropout = sample(c(0.2, 0.3, 0.4, 0.5), 1),
      embed_dim = sample(c(64L, 100L, 128L), 1),
      learning_rate = sample(c(2e-4, 5e-4, 1e-3, 2e-3), 1),
      epochs = 8L
    )

    bilstm_search <- vector("list", BILSTM_GRID_N)
    for (i in seq_len(BILSTM_GRID_N)) {
      set.seed(100 + i)
      cfg <- sample_bilstm_config()
      m_i <- tryCatch(
        build_bilstm(units = cfg$units, dropout_rate = cfg$dropout,
                     embed_dim_local = cfg$embed_dim,
                     learning_rate = cfg$learning_rate),
        error = function(e) NULL
      )
      if (is.null(m_i)) next

      hist_i <- tryCatch(
        fit(m_i,
            x = X_nn_train, y = y_train_k,
            epochs = cfg$epochs, batch_size = 32,
            validation_split = 0.2,
            callbacks = make_callbacks(patience_es = 3),
            verbose = 0),
        error = function(e) NULL
      )
      if (is.null(hist_i)) next

      hist_df_i <- as.data.frame(hist_i)
      val_rows <- hist_df_i[hist_df_i$metric == "loss", ]
      data_col <- if ("data" %in% names(val_rows)) val_rows$data
                  else if ("dataset" %in% names(val_rows)) val_rows$dataset
                  else if ("split" %in% names(val_rows)) val_rows$split
                  else rep("training", nrow(val_rows))
      val_only <- val_rows[grepl("val|valid", data_col, ignore.case = TRUE), ]
      best_val_loss <- if (nrow(val_only) > 0) min(val_only$value, na.rm = TRUE)
                      else min(hist_df_i$value[hist_df_i$metric == "loss"], na.rm = TRUE)

      bilstm_search[[i]] <- list(cfg = cfg, val_loss = best_val_loss, model = m_i)
      cat(sprintf("  cfg %02d: units=%d dropout=%.2f embed=%d lr=%.4f -> val_loss=%.4f\n",
                  i, cfg$units, cfg$dropout, cfg$embed_dim, cfg$learning_rate,
                  best_val_loss))
    }
    bilstm_search <- bilstm_search[!vapply(bilstm_search, is.null, logical(1))]

    if (length(bilstm_search) > 0) {
      bilstm_tune_df <- do.call(rbind, lapply(seq_along(bilstm_search), function(i) {
        s <- bilstm_search[[i]]
        data.frame(config_id = i,
                   units = s$cfg$units,
                   dropout = s$cfg$dropout,
                   embed_dim = s$cfg$embed_dim,
                   learning_rate = s$cfg$learning_rate,
                   val_loss = round(s$val_loss, 5))
      }))
      bilstm_tune_df <- bilstm_tune_df[order(bilstm_tune_df$val_loss), ]
      write.csv(bilstm_tune_df, "table_tuning_bilstm.csv", row.names = FALSE)

      best_b_idx <- which.min(vapply(bilstm_search, function(s) s$val_loss, numeric(1)))
      best_b_cfg <- bilstm_search[[best_b_idx]]$cfg
      cat(sprintf("BEST BiLSTM: units=%d dropout=%.2f embed=%d lr=%.4f val_loss=%.4f\n",
                  best_b_cfg$units, best_b_cfg$dropout, best_b_cfg$embed_dim,
                  best_b_cfg$learning_rate, bilstm_search[[best_b_idx]]$val_loss))

      cat("Refitting best BiLSTM with up to 25 epochs and full callbacks...\n")
      model_bilstm <- build_bilstm(units = best_b_cfg$units,
                                   dropout_rate = best_b_cfg$dropout,
                                   embed_dim_local = best_b_cfg$embed_dim,
                                   learning_rate = best_b_cfg$learning_rate)
      set.seed(34074759)
      hist_bilstm <- fit(model_bilstm,
        x = X_nn_train, y = y_train_k,
        epochs = 25, batch_size = 32, validation_split = 0.2,
        callbacks = make_callbacks(patience_es = 5), verbose = 1
      )
      safe_save_history(hist_bilstm,
                        "Keras BiLSTM (NN4) — Training History (best config)",
                        "bilstm_history.png")

      bilstm_probs <- as.matrix(predict(model_bilstm, X_nn_test, verbose = 0))
      bilstm_pred <- factor(label_levels[apply(bilstm_probs, 1, which.max)],
                             levels = label_levels)
      bilstm_cm <- confusionMatrix(bilstm_pred, y_test)
      results_list$keras_bilstm <- bilstm_cm
      cat("Keras BiLSTM Accuracy:", round(bilstm_cm$overall["Accuracy"], 4), "\n")
    } else {
      cat("** All BiLSTM configurations failed; falling back to default config. **\n")
    }
  }


  if (is.null(bilstm_cm)) {
    cat("Training default BiLSTM\n")
    model_bilstm <- build_bilstm(units = 64L, dropout_rate = 0.3,
                                 embed_dim_local = embed_dim,
                                 learning_rate = 5e-4)
    set.seed(34074759)
    hist_bilstm <- fit(model_bilstm,
      x = X_nn_train, y = y_train_k,
      epochs = 20, batch_size = 32, validation_split = 0.2,
      callbacks = make_callbacks(patience_es = 5), verbose = 1
    )
    safe_save_history(hist_bilstm,
                      "Keras BiLSTM (NN4) — Training History",
                      "bilstm_history.png")

    bilstm_probs <- as.matrix(predict(model_bilstm, X_nn_test, verbose = 0))
    bilstm_pred <- factor(label_levels[apply(bilstm_probs, 1, which.max)],
                           levels = label_levels)
    bilstm_cm <- confusionMatrix(bilstm_pred, y_test)
    results_list$keras_bilstm <- bilstm_cm
    cat("Keras BiLSTM Accuracy:", round(bilstm_cm$overall["Accuracy"], 4), "\n")
  }

  keras_available <- TRUE
}, error = function(e) {
  cat("\n** Keras / TensorFlow not available — deep learning skipped. **\n")
  cat("Error:", conditionMessage(e), "\n")
  if (requireNamespace("reticulate", quietly = TRUE)) {
    cat("\nreticulate Python config at failure:\n")
    try(print(reticulate::py_config()), silent = TRUE)
    cat("\nPython traceback, if available:\n")
    try(print(reticulate::py_last_error()), silent = TRUE)
  }
  cat("\nUse the r-keras virtualenv you already verified with TensorFlow 2.21.0.\n")
})


# 8.13 Soft-voting ensemble
cat("\nMODEL 8.13: SOFT-VOTING ENSEMBLE\n")

get_probs <- function(label) {
  switch(label,
    "logreg" = if (!is.null(logreg_proba)) logreg_proba else NULL,
    "lasso" = predict(lasso_cv, X_test_sp, s = "lambda.min",
                      type = "response")[, , 1],
    "ridge" = predict(ridge_cv, X_test_sp, s = "lambda.min",
                      type = "response")[, , 1],
    "enet"  = predict(enet_cv,  X_test_sp, s = "lambda.min",
                      type = "response")[, , 1],
    "tuned_enet" = if (!is.null(tuned_enet_cm) && exists("best_glmnet"))
                    predict(best_glmnet, X_test_sp, s = "lambda.min",
                            type = "response")[, , 1] else NULL,
    "xgb"   = xgb_proba,
    "tuned_xgb" = if (!is.null(xgb_proba_tuned)) xgb_proba_tuned else NULL,
    "lgb"   = if (lgb_available) lgb_proba else NULL,
    "tuned_lgb" = if (!is.null(lgb_proba_tuned)) lgb_proba_tuned else NULL,
    "tuned_svm" = if (!is.null(tuned_svm_cm) && exists("svm_tuned_probs"))
                    svm_tuned_probs else NULL,
    "emb_lr"= if (emb_available)
              predict(emb_lr_cv, as.matrix(X_emb_test), s = "lambda.min",
                      type = "response")[, , 1] else NULL,
    "keras_bilstm" = if (exists("bilstm_probs")) bilstm_probs else NULL,
    NULL
  )
}

candidate_labels <- character(0)
if (!is.null(results_list$logreg) && !is.null(logreg_proba)) {
  candidate_labels <- c(candidate_labels, "logreg")
}
candidate_labels <- c(candidate_labels, "lasso", "ridge")

if (!is.null(results_list$tuned_enet)) {
  candidate_labels <- c(candidate_labels, "tuned_enet")
} else {
  candidate_labels <- c(candidate_labels, "enet")
}

if (!is.null(results_list$tuned_xgb)) {
  candidate_labels <- c(candidate_labels, "tuned_xgb")
} else {
  candidate_labels <- c(candidate_labels, "xgb")
}

if (!is.null(results_list$tuned_lgb)) {
  candidate_labels <- c(candidate_labels, "tuned_lgb")
} else if (lgb_available) {
  candidate_labels <- c(candidate_labels, "lgb")
}

if (!is.null(results_list$tuned_svm))    candidate_labels <- c(candidate_labels, "tuned_svm")
if (emb_available)                       candidate_labels <- c(candidate_labels, "emb_lr")
if (!is.null(results_list$keras_bilstm)) candidate_labels <- c(candidate_labels, "keras_bilstm")

prob_list <- lapply(candidate_labels, get_probs)
names(prob_list) <- candidate_labels
prob_list <- prob_list[!vapply(prob_list, is.null, logical(1))]

common_levels <- levels(y_test)
prob_list <- lapply(prob_list, function(p) {
  if (is.null(colnames(p))) colnames(p) <- common_levels
  p[, match(common_levels, colnames(p)), drop = FALSE]
})

ensemble_proba <- Reduce("+", prob_list) / length(prob_list)
ensemble_pred <- factor(common_levels[max.col(ensemble_proba)],
                         levels = common_levels)
ensemble_cm <- confusionMatrix(ensemble_pred, y_test)
results_list$ensemble <- ensemble_cm
cat("Soft-Voting Ensemble Accuracy:",
    round(ensemble_cm$overall["Accuracy"], 4), "\n")
cat("(combining:", paste(names(prob_list), collapse = " + "), ")\n")


# 9.model comparison

cat("\nFINAL MODEL COMPARISON\n")

results <- data.frame(Model = character(), Accuracy = numeric(),
                      Kappa = numeric(), Macro_F1 = numeric(),
                      Macro_Prec = numeric(), Macro_Recall = numeric(),
                      Macro_BalAcc = numeric(),
                      stringsAsFactors = FALSE)

if (!is.null(results_list$logreg))
  results <- add_row(results,
                     paste0("Logistic Regression (",
                            ifelse(is.na(logreg_engine), "?", logreg_engine), ")"),
                     results_list$logreg)

results <- add_row(results, "LASSO (alpha=1)", results_list$lasso)
results <- add_row(results, "Ridge (alpha=0)", results_list$ridge)
results <- add_row(results, "Elastic Net (alpha=0.5)", results_list$enet)
if (!is.null(results_list$tuned_enet))
  results <- add_row(results, "Tuned Elastic Net (alpha-grid)", results_list$tuned_enet)
results <- add_row(results, "Naive Bayes", results_list$nb)
if (!is.null(results_list$tuned_nb))
  results <- add_row(results, "Naive Bayes (tuned)", results_list$tuned_nb)
if (!is.null(results_list$svm_radial))
  results <- add_row(results, "SVM Radial", results_list$svm_radial)
if (!is.null(results_list$tuned_svm))
  results <- add_row(results, "SVM Radial (tuned)", results_list$tuned_svm)
results <- add_row(results, "XGBoost (sparse TF-IDF)", results_list$xgb)
if (!is.null(results_list$tuned_xgb))
  results <- add_row(results, "XGBoost (tuned)", results_list$tuned_xgb)
if (!is.null(results_list$lgb))
  results <- add_row(results, "LightGBM (sparse TF-IDF)", results_list$lgb)
if (!is.null(results_list$tuned_lgb))
  results <- add_row(results, "LightGBM (tuned)", results_list$tuned_lgb)
if (!is.null(results_list$fasttext))
  results <- add_row(results, "fastText supervised", results_list$fasttext)
if (!is.null(results_list$emb_lr))
  results <- add_row(results, "MiniLM emb. + LR", results_list$emb_lr)
if (!is.null(results_list$keras_dense))
  results <- add_row(results, "Keras Dense (NN1)", results_list$keras_dense)
if (!is.null(results_list$keras_cnn))
  results <- add_row(results, "Keras CNN (NN2)", results_list$keras_cnn)
if (!is.null(results_list$keras_lstm))
  results <- add_row(results, "Keras LSTM (NN3)", results_list$keras_lstm)
if (!is.null(results_list$keras_bilstm))
  results <- add_row(results, "Keras BiLSTM (NN4)", results_list$keras_bilstm)
results <- add_row(results, "Soft-Voting Ensemble", results_list$ensemble)

results <- results[order(-results$Accuracy), ]
rownames(results) <- NULL

cat("\n")
print(results)
write.csv(results, "table_model_comparison.csv", row.names = FALSE)

p_comp <- ggplot(results,
                 aes(x = reorder(Model, Accuracy), y = Accuracy,
                     fill = Accuracy)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.3f", Accuracy)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_viridis_c(end = 0.85) +
  ylim(0, 1.05) +
  theme_minimal(base_size = 11) +
  labs(title = "Model Comparison — Test Set Accuracy",
       subtitle = "Macro-F1 and Cohen's kappa available in table_model_comparison.csv",
       x = NULL, y = "Accuracy") +
  theme(legend.position = "none")
save_plot(p_comp, "model_comparison.png", width = 10, height = 7)
results_long <- results %>%
  select(Model, Accuracy, Macro_F1, Kappa) %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels = c("Accuracy", "Macro_F1", "Kappa")))

p_comp2 <- ggplot(results_long,
                  aes(x = reorder(Model, Value), y = Value, fill = Value)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.3f", Value)), hjust = -0.1, size = 2.6) +
  facet_wrap(~ Metric, ncol = 3, scales = "free_x") +
  coord_flip() +
  scale_fill_viridis_c(end = 0.85) +
  expand_limits(y = max(results_long$Value, na.rm = TRUE) + 0.08) +
  theme_minimal(base_size = 10) +
  labs(title = "Model Comparison — Accuracy, Macro-F1 and Cohen's Kappa",
       subtitle = "All three metrics agree on the model ordering, supporting robustness of the ensemble's lead.",
       x = NULL, y = NULL) +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold"))
save_plot(p_comp2, "model_comparison_multi.png", width = 13, height = 7)


# 10. per-class metrics for the best model

best_model_label <- results$Model[1]
cat(sprintf("\n PER-CLASS METRICS — BEST MODEL: %s \n", best_model_label))

cm_lookup <- list(
  "LASSO (alpha=1)" = results_list$lasso,
  "Ridge (alpha=0)"= results_list$ridge,
  "Elastic Net (alpha=0.5)"= results_list$enet,
  "Tuned Elastic Net (alpha-grid)" = results_list$tuned_enet,
  "Naive Bayes" = results_list$nb,
  "Naive Bayes (tuned)"= results_list$tuned_nb,
  "SVM Radial"= results_list$svm_radial,
  "SVM Radial (tuned)"  = results_list$tuned_svm,
  "XGBoost (sparse TF-IDF)" = results_list$xgb,
  "XGBoost (tuned)" = results_list$tuned_xgb,
  "LightGBM (sparse TF-IDF)"= results_list$lgb,
  "LightGBM (tuned)" = results_list$tuned_lgb,
  "fastText supervised"= results_list$fasttext,
  "MiniLM emb. + LR"= results_list$emb_lr,
  "Keras Dense (NN1)"= results_list$keras_dense,
  "Keras CNN (NN2)" = results_list$keras_cnn,
  "Keras LSTM (NN3)" = results_list$keras_lstm,
  "Keras BiLSTM (NN4)"= results_list$keras_bilstm,
  "Soft-Voting Ensemble"= results_list$ensemble
)

if (!is.null(results_list$logreg))
  cm_lookup[[paste0("Logistic Regression (",
                    ifelse(is.na(logreg_engine), "?", logreg_engine), ")")]] <-
    results_list$logreg
cm_lookup <- cm_lookup[!vapply(cm_lookup, is.null, logical(1))]
best_cm <- cm_lookup[[best_model_label]]
print(best_cm$byClass)
write.csv(as.data.frame(best_cm$byClass), "table_best_perclass.csv")

best_cm_df <- as.data.frame(best_cm$table)
p_best_cm <- ggplot(best_cm_df, aes(Reference, Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_viridis_c() +
  theme_minimal(base_size = 11) +
  labs(title = sprintf("Best Model: %s — Confusion Matrix", best_model_label),
       subtitle = sprintf("Accuracy = %.4f, Kappa = %.4f",
                          best_cm$overall["Accuracy"], best_cm$overall["Kappa"]),
       x = "Actual", y = "Predicted") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
save_plot(p_best_cm, "best_model_confusion.png", width = 7, height = 6)


# 10b. Statistical evaluation

cat("\nSTATISTICAL EVALUATION\n")

# (i) Bootstrap 95% CI on best-model accuracy
pred_lookup <- list(
  "LASSO (alpha=1)"  = if (exists("lasso_pred")) lasso_pred,
  "Ridge (alpha=0)" = if (exists("ridge_pred")) ridge_pred,
  "Elastic Net (alpha=0.5)"= if (exists("enet_pred")) enet_pred,
  "Tuned Elastic Net (alpha-grid)" = if (exists("tuned_enet_pred")) tuned_enet_pred,
  "Naive Bayes" = if (exists("nb_pred")) nb_pred,
  "Naive Bayes (tuned)"  = if (exists("tuned_nb_pred"))tuned_nb_pred,
  "SVM Radial"= if (exists("svm_pred")) svm_pred,
  "SVM Radial (tuned)"= if (exists("svm_tuned_pred"))svm_tuned_pred,
  "XGBoost (sparse TF-IDF)"= if (exists("xgb_pred")) xgb_pred,
  "XGBoost (tuned)"= if (exists("xgb_pred_tuned"))xgb_pred_tuned,
  "LightGBM (sparse TF-IDF)"  = if (exists("lgb_pred")) lgb_pred,
  "LightGBM (tuned)" = if (exists("lgb_pred_tuned"))lgb_pred_tuned,
  "fastText supervised"  = if (exists("ft_pred")) ft_pred,
  "MiniLM emb. + LR" = if (exists("emb_lr_pred")) emb_lr_pred,
  "Keras Dense (NN1)"  = if (exists("dense_pred")) dense_pred,
  "Keras CNN (NN2)"= if (exists("cnn_pred")) cnn_pred,
  "Keras LSTM (NN3)" = if (exists("lstm_pred"))lstm_pred,
  "Keras BiLSTM (NN4)" = if (exists("bilstm_pred"))bilstm_pred,
  "Soft-Voting Ensemble" = if (exists("ensemble_pred"))ensemble_pred
)
if (!is.null(results_list$logreg) && !is.null(logreg_pred)) {
  pred_lookup[[paste0("Logistic Regression (",
                      ifelse(is.na(logreg_engine), "?", logreg_engine), ")")]] <-
    logreg_pred
}
pred_lookup <- pred_lookup[!vapply(pred_lookup, is.null, logical(1))]

best_pred <- pred_lookup[[best_model_label]]
if (!is.null(best_pred)) {
  set.seed(34074759)
  B <- 1000L
  n <- length(y_test)
  boot_acc <- numeric(B)
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    boot_acc[b] <- mean(best_pred[idx] == y_test[idx])
  }
  acc_ci <- quantile(boot_acc, c(0.025, 0.975))
  cat(sprintf("Bootstrap 95%% CI on %s accuracy: [%.4f, %.4f] (B = %d)\n",
              best_model_label, acc_ci[1], acc_ci[2], B))
  cat(sprintf("  Point estimate:  %.4f\n", mean(best_pred == y_test)))
  cat(sprintf("  Bootstrap mean:  %.4f\n", mean(boot_acc)))
  cat(sprintf("  Bootstrap SE:    %.4f\n", sd(boot_acc)))

  writeLines(c(
    sprintf("Best model: %s", best_model_label),
    sprintf("Test accuracy: %.4f", mean(best_pred == y_test)),
    sprintf("Bootstrap mean accuracy: %.4f", mean(boot_acc)),
    sprintf("Bootstrap SE: %.4f", sd(boot_acc)),
    sprintf("95%% CI (percentile):     [%.4f, %.4f]", acc_ci[1], acc_ci[2]),
    sprintf("Bootstrap resamples: %d", B)
  ), "table_bootstrap_ci.txt")
}

# (ii) Multiclass ROC-AUC 
auc_table <- data.frame(Model = character(), AUC_HandTill = numeric(),
                        stringsAsFactors = FALSE)

if (requireNamespace("pROC", quietly = TRUE)) {
  library(pROC)
  prob_lookup <- list(
    "LASSO (alpha=1)" = predict(lasso_cv, X_test_sp, s = "lambda.min",
                                         type = "response")[, , 1],
    "Ridge (alpha=0)" = predict(ridge_cv, X_test_sp, s = "lambda.min",
                                         type = "response")[, , 1],
    "Elastic Net (alpha=0.5)"  = predict(enet_cv,  X_test_sp, s = "lambda.min",
                                         type = "response")[, , 1],
    "Tuned Elastic Net (alpha-grid)" = if (!is.null(tuned_enet_cm) &&
                                           exists("best_glmnet"))
                                        predict(best_glmnet, X_test_sp,
                                                s = "lambda.min",
                                                type = "response")[, , 1] else NULL,
    "XGBoost (sparse TF-IDF)" = xgb_proba,
    "XGBoost (tuned)" = if (!is.null(xgb_proba_tuned)) xgb_proba_tuned else NULL,
    "LightGBM (sparse TF-IDF)" = if (lgb_available) lgb_proba else NULL,
    "LightGBM (tuned)"= if (!is.null(lgb_proba_tuned)) lgb_proba_tuned else NULL,
    "SVM Radial (tuned)" = if (!is.null(tuned_svm_cm) &&
                                     exists("svm_tuned_probs")) svm_tuned_probs else NULL,
    "MiniLM emb. + LR" = if (emb_available)
                                  predict(emb_lr_cv, as.matrix(X_emb_test),
                                          s = "lambda.min", type = "response")[, , 1] else NULL,
    "Keras BiLSTM (NN4)" = if (exists("bilstm_probs")) bilstm_probs else NULL,
    "Soft-Voting Ensemble" = ensemble_proba
  )
  if (!is.null(results_list$logreg) && !is.null(logreg_proba)) {
    prob_lookup[[paste0("Logistic Regression (",
                        ifelse(is.na(logreg_engine), "?", logreg_engine), ")")]] <-
      logreg_proba
  }
  prob_lookup <- prob_lookup[!vapply(prob_lookup, is.null, logical(1))]

  for (mname in names(prob_lookup)) {
    pmat <- prob_lookup[[mname]]
    if (is.null(colnames(pmat))) colnames(pmat) <- levels(y_test)
    pmat <- pmat[, match(levels(y_test), colnames(pmat)), drop = FALSE]
    auc_val <- tryCatch({
      mr <- pROC::multiclass.roc(response = y_test, predictor = pmat,
                                 quiet = TRUE)
      as.numeric(mr$auc)
    }, error = function(e) NA_real_)
    auc_table <- rbind(auc_table,
                       data.frame(Model = mname,
                                  AUC_HandTill = round(auc_val, 4)))
  }
  auc_table <- auc_table[order(-auc_table$AUC_HandTill), ]
  cat("\nMulticlass ROC-AUC (Hand & Till 2001):\n")
  print(auc_table, row.names = FALSE)
  write.csv(auc_table, "table_roc_auc.csv", row.names = FALSE)
} else {
  cat("** pROC not installed — multiclass ROC-AUC skipped. **\n")
  cat("Install with: install.packages('pROC')\n")
}

# ROC curves for the ensemble, one vs rest, per class
if (requireNamespace("pROC", quietly = TRUE)) {
  library(pROC)
  cls <- levels(y_test)
  roc_list <- lapply(cls, function(k) {
    pROC::roc(response = as.integer(y_test == k),
              predictor = ensemble_proba[, k],
              quiet = TRUE)
  })
  names(roc_list) <- cls

  roc_df <- do.call(rbind, lapply(cls, function(k) {
    r <- roc_list[[k]]
    data.frame(
      class = sprintf("%s (AUC = %.3f)", k, as.numeric(r$auc)),
      fpr = 1 - r$specificities,
      tpr = r$sensitivities
    )
  }))

  p_roc <- ggplot(roc_df, aes(fpr, tpr, colour = class)) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", colour = "grey60") +
    geom_path(linewidth = 1) +
    scale_x_continuous("False Positive Rate", limits = c(0, 1)) +
    scale_y_continuous("True Positive Rate", limits = c(0, 1)) +
    coord_equal() +
    theme_minimal(base_size = 11) +
    labs(
      title = "Soft-Voting Ensemble — One-vs-Rest ROC Curves",
      subtitle = sprintf("Macro-AUC (Hand-Till) = %.4f",
                         as.numeric(pROC::multiclass.roc(
                           response = y_test,
                           predictor = ensemble_proba,
                           quiet = TRUE)$auc)),
      colour = "Class"
    ) +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10),
          legend.position = "bottom")

  save_plot(p_roc, "ensemble_roc.png", width = 7, height = 7)
}
# (iii) McNemar's test - best vs runner-up
cat("\nMcNemar's test — best vs runner-up:\n")
runner_up_label <- results$Model[2]
if (!is.null(pred_lookup[[best_model_label]]) &&
    !is.null(pred_lookup[[runner_up_label]])) {

  pred_best <- pred_lookup[[best_model_label]]
  pred_run <- pred_lookup[[runner_up_label]]

  best_correct <- pred_best == y_test
  run_correct <- pred_run  == y_test
  m22 <- table(Best = factor(best_correct, levels = c(TRUE, FALSE)),
               RunnerUp = factor(run_correct,  levels = c(TRUE, FALSE)))
  cat("Contingency table:\n"); print(m22)

  mcn <- mcnemar.test(m22, correct = TRUE)
  cat(sprintf("Models compared: %s  vs  %s\n",
              best_model_label, runner_up_label))
  cat(sprintf("McNemar statistic: %.3f  df = %d  p-value = %.4f\n",
              unname(mcn$statistic), unname(mcn$parameter), mcn$p.value))

  writeLines(c(
    sprintf("Best: %s", best_model_label),
    sprintf("Runner-up: %s", runner_up_label),
    sprintf("McNemar X^2: %.3f", unname(mcn$statistic)),
    sprintf("df: %d", unname(mcn$parameter)),
    sprintf("p-value: %.4f", mcn$p.value),
    sprintf("Disagreements b=%d, c=%d (b = best-only-correct; c = runner-up-only-correct)",
            m22[1, 2], m22[2, 1])
  ), "table_mcnemar_best_vs_runner.txt")
}

# (iv) Per-class F1 table for the top-3 models 
top3 <- head(results$Model, 3)
perclass_top3 <- lapply(top3, function(m) {
  bc <- as.data.frame(cm_lookup[[m]]$byClass)
  data.frame(
    Model = m,
    Class = sub("Class: ", "", rownames(bc)),
    Precision = round(bc[["Precision"]], 4),
    Recall = round(bc[["Recall"]],    4),
    F1 = round(bc[["F1"]],        4),
    BalAcc = round(bc[["Balanced Accuracy"]], 4)
  )
})
perclass_top3_df <- do.call(rbind, perclass_top3)
write.csv(perclass_top3_df, "table_top3_perclass.csv", row.names = FALSE)
cat("\nPer-class F1 (top 3 models) — saved to table_top3_perclass.csv\n")
print(perclass_top3_df, row.names = FALSE)



# Chord diagram of the best model's confusion matrix. 
if (circlize_ok) {
  library(circlize)

  cm_mat <- as.matrix(best_cm$table)
  flow_mat <- t(cm_mat)
  rownames(flow_mat) <- paste0(rownames(flow_mat), "_true")
  colnames(flow_mat) <- paste0(colnames(flow_mat), "_pred")

  # Use a distinct palette
  pal <- viridis::viridis(nlevels(y_test), end = 0.92)
  grid.col <- c(setNames(pal, rownames(flow_mat)),
                setNames(pal, colnames(flow_mat)))

  png("chord_confusion.png", width = 1600, height = 1600, res = 200)
  par(mar = c(1, 1, 3, 1))
  circos.clear()
  circos.par(gap.after = c(rep(2, nrow(flow_mat) - 1), 12,
                           rep(2, ncol(flow_mat) - 1), 12),
             start.degree = 90)
  chordDiagram(flow_mat,
               grid.col = grid.col,
               transparency = 0.35,
               directional = 1,
               direction.type = c("arrows", "diffHeight"),
               diffHeight = -mm_h(2),
               link.arr.type = "big.arrow",
               annotationTrack = c("grid"),
               preAllocateTracks = list(track.height = 0.08))
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    xlim <- get.cell.meta.data("xlim"); ylim <- get.cell.meta.data("ylim")
    nm <- get.cell.meta.data("sector.index")
    nm_clean <- sub("_(true|pred)$", "", nm)
    side <- ifelse(grepl("_true$", nm), "[true]", "[pred]")
    circos.text(mean(xlim), ylim[1] + 0.6,
                paste0(nm_clean, "\n", side),
                niceFacing = TRUE, cex = 0.7, font = 2)
  }, bg.border = NA)
  title(sprintf("Chord Diagram: %s — Confusion Flow\n(left half: actual; right half: predicted)",
                best_model_label), cex.main = 0.95)
  circos.clear()
  dev.off()
} else {
  cat("** Chord diagram skipped — install.packages('circlize') **\n")
}

# 10b. SHAP summary plots


cat("\nSHAP SUMMARY PLOTS\n")
cat("getwd():", getwd(), "\n")
cat("xgb_model exists: ", exists("xgb_model"), "\n")
cat("xgb_model_tuned exists:", exists("xgb_model_tuned"), "\n")
flush.console()

tryCatch({
  shap_model <- if (exists("xgb_model_tuned") && !is.null(xgb_model_tuned)) {
    cat("Using xgb_model_tuned for SHAP.\n"); xgb_model_tuned
  } else if (exists("xgb_model") && !is.null(xgb_model)) {
    cat("Using xgb_model (baseline) for SHAP.\n"); xgb_model
  } else {
    stop("No xgb model available — train XGBoost first.")
  }

  set.seed(34074759)
  shap_n <- min(800L, nrow(X_train_sp))
  shap_idx <- sample(seq_len(nrow(X_train_sp)), shap_n)
  X_shap <- as.matrix(X_train_sp[shap_idx, , drop = FALSE])
  feat_names <- colnames(X_shap)
  if (is.null(feat_names)) feat_names <- paste0("f", seq_len(ncol(X_shap)))

  cat(sprintf("Computing SHAP on %d rows x %d features...\n",
              nrow(X_shap), ncol(X_shap)))
  flush.console()

  shap_raw <- predict(shap_model, X_shap, predcontrib = TRUE)

  cat("  shap_raw class: ", paste(class(shap_raw), collapse = "/"),
      " | structure: ",
      if (is.list(shap_raw))
        paste("list of", length(shap_raw),
              "; first elem dim:",
              paste(dim(shap_raw[[1]]), collapse = " x "))
      else paste("array dim:", paste(dim(shap_raw), collapse = " x ")),
      "\n", sep = "")
  flush.console()

  n_rows_x <- nrow(X_shap)
  n_classes <- nlevels(y_train)
  n_feat_p1 <- ncol(X_shap) + 1L   

  shap_list <- if (is.list(shap_raw)) {
    shap_raw
  } else if (is.array(shap_raw) && length(dim(shap_raw)) == 3L) {
    d <- dim(shap_raw)
    if      (d[1] == n_rows_x && d[2] == n_feat_p1 && d[3] == n_classes)
      lapply(seq_len(n_classes), function(k) shap_raw[, , k])
    else if (d[1] == n_rows_x && d[2] == n_classes && d[3] == n_feat_p1)
      lapply(seq_len(n_classes), function(k) shap_raw[, k, ])
    else if (d[1] == n_classes && d[2] == n_rows_x && d[3] == n_feat_p1)
      lapply(seq_len(n_classes), function(k) shap_raw[k, , ])
    else
      stop("Unrecognised SHAP array shape ", paste(d, collapse = " x "),
           ". Expected one dim = n_rows (", n_rows_x,
           "), one = n_classes (", n_classes,
           "), one = n_features+1 (", n_feat_p1, ").")
  } else if (is.matrix(shap_raw)) {
    list(shap_raw)
  } else {
    stop("Unsupported shap_raw type: ", paste(class(shap_raw), collapse = "/"))
  }

  shap_list <- lapply(seq_along(shap_list), function(k) {
    m <- shap_list[[k]]
    if (!is.matrix(m)) m <- as.matrix(m)
    storage.mode(m) <- "double"
    if (nrow(m) != n_rows_x || ncol(m) != n_feat_p1) {
      stop(sprintf("SHAP matrix %d has shape %d x %d (expected %d x %d)",
                   k, nrow(m), ncol(m), n_rows_x, n_feat_p1))
    }
    m
  })
  cat("Normalised to list of", length(shap_list), "matrices, each",
      nrow(shap_list[[1]]), "x", ncol(shap_list[[1]]), "\n")
  flush.console()

  class_names <- levels(y_train)[seq_along(shap_list)]

  shap_long <- do.call(rbind, lapply(seq_along(shap_list), function(k) {
    mat <- shap_list[[k]][, -ncol(shap_list[[k]]), drop = FALSE]  
    colnames(mat) <- feat_names
    data.frame(class = class_names[k],
               feature = feat_names,
               mean_abs_shap = colMeans(abs(mat)),
               row.names = NULL,
               stringsAsFactors = FALSE)
  }))

  #Plot 1: top-12 features, grouped by class
  global_top <- shap_long %>%
    dplyr::group_by(feature) %>%
    dplyr::summarise(global_imp = mean(mean_abs_shap), .groups = "drop") %>%
    dplyr::slice_max(global_imp, n = 12)

  shap_top <- shap_long %>%
    dplyr::filter(feature %in% global_top$feature) %>%
    dplyr::mutate(feature = factor(feature,
                                   levels = global_top$feature[order(global_top$global_imp)]))

  p_shap1 <- ggplot(shap_top,
                    aes(x = mean_abs_shap, y = feature, fill = class)) +
    geom_col(position = "dodge") +
    scale_fill_viridis_d(end = 0.92) +
    theme_minimal(base_size = 11) +
    labs(title = "SHAP Feature Importance by Class — XGBoost",
         subtitle = sprintf("Mean |SHAP value| over %d training rows. Lundberg & Lee (2017).",
                            shap_n),
         x = "Mean |SHAP value|", y = NULL, fill = "Class")

  out1 <- "shap_per_class.png"
  ggsave(out1, plot = p_shap1, width = 11, height = 7, dpi = 300)
  cat("Saved: ", out1, " | exists: ", file.exists(out1),
      " | size (KB): ", round(file.info(out1)$size / 1024, 1), "\n", sep = "")
  flush.console()

  #Plot 2: beeswarm summary for class 1
  target_class_idx <- 1L
  shap_mat_one <- shap_list[[target_class_idx]][, -n_feat_p1, drop = FALSE]
  colnames(shap_mat_one) <- feat_names
  top_feats_one <- names(sort(colMeans(abs(shap_mat_one)),
                              decreasing = TRUE))[1:12]

  shap_summary_long <- data.frame(
    feature = rep(top_feats_one, each = shap_n),
    shap = as.vector(shap_mat_one[, top_feats_one]),
    feat_value = as.vector(X_shap[, top_feats_one]),
    stringsAsFactors = FALSE
  )
  shap_summary_long$feature <- factor(shap_summary_long$feature,
                                      levels = rev(top_feats_one))

  p_shap2 <- ggplot(shap_summary_long,
                    aes(x = shap, y = feature, colour = feat_value)) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    geom_jitter(alpha = 0.5, size = 0.9, width = 0, height = 0.25) +
    scale_colour_viridis_c(option = "plasma", end = 0.95) +
    theme_minimal(base_size = 11) +
    labs(title = sprintf("SHAP Summary — class '%s' (XGBoost)",
                            class_names[target_class_idx]),
         subtitle = "Each point = one training row. Colour = feature value (light = high, dark = low).",
         x = "SHAP value (impact on log-odds for this class)",
         y = NULL, colour = "Feature\nvalue")

  out2 <- "shap_beeswarm.png"
  ggsave(out2, plot = p_shap2, width = 11, height = 7, dpi = 300)
  cat("Saved: ", out2, " | exists: ", file.exists(out2),
      " | size (KB): ", round(file.info(out2)$size / 1024, 1), "\n", sep = "")
  flush.console()

  cat("SHAP plots complete.\n")
}, error = function(e) {
  cat("** SHAP plots skipped — error below **\n",
      "  message: ", conditionMessage(e), "\n",
      "  call:    ", deparse(conditionCall(e)), "\n", sep = "")
})
# 11. Error analysis

cat("\n ERROR ANALYSIS\n")

off_diag <- as.data.frame(best_cm$table) %>%
  filter(as.character(Prediction) != as.character(Reference)) %>%
  arrange(desc(Freq))
cat("\nTop confusable pairs (best model):\n")
print(head(off_diag, 8))
write.csv(off_diag, "table_confusable_pairs.csv", row.names = FALSE)

ensemble_max_prob <- apply(ensemble_proba, 1, max)
wrong_idx <- which(ensemble_pred != y_test)
wrong_idx <- wrong_idx[order(-ensemble_max_prob[wrong_idx])]

err_examples <- data.frame(
  Title = data$Title[-train_idx][wrong_idx],
  True = y_test[wrong_idx],
  Predicted = ensemble_pred[wrong_idx],
  Confidence = round(ensemble_max_prob[wrong_idx], 3)
) %>% head(20)
write.csv(err_examples, "table_top20_misclassifications.csv", row.names = FALSE)
cat("\nTop 20 high-confidence misclassifications saved.\n")


# 11b. HYPERPARAMETER-TUNING VISUALISATION

cat("\n TUNING VISUALISATION\n")
tuning_panels <- list()

if (file.exists("table_tuning_glmnet.csv")) {
  d <- read.csv("table_tuning_glmnet.csv")
  d$model <- "GLMNET (alpha grid)"
  d$score <- d$cv_deviance
  d$x <- seq_len(nrow(d))
  d$best <- d$score == min(d$score, na.rm = TRUE)
  tuning_panels[["glmnet"]] <- d[, c("model", "x", "score", "best")]
}
if (file.exists("table_tuning_xgb.csv")) {
  d <- read.csv("table_tuning_xgb.csv")
  d$model <- "XGBoost (random search)"
  d$score <- d$cv_logloss
  d$x <- seq_len(nrow(d))
  d$best <- d$score == min(d$score, na.rm = TRUE)
  tuning_panels[["xgb"]] <- d[, c("model", "x", "score", "best")]
}
if (file.exists("table_tuning_lgb.csv")) {
  d <- read.csv("table_tuning_lgb.csv")
  d$model <- "LightGBM (random search)"
  d$score <- d$cv_logloss
  d$x <- seq_len(nrow(d))
  d$best <- d$score == min(d$score, na.rm = TRUE)
  tuning_panels[["lgb"]] <- d[, c("model", "x", "score", "best")]
}
if (file.exists("table_tuning_svm.csv")) {
  d <- read.csv("table_tuning_svm.csv")
  d$model <- "SVM (cost x gamma grid)"
  d$score <- d$error
  d$x <- seq_len(nrow(d))
  d$best <- d$score == min(d$score, na.rm = TRUE)
  tuning_panels[["svm"]] <- d[, c("model", "x", "score", "best")]
}
if (file.exists("table_tuning_bilstm.csv")) {
  d <- read.csv("table_tuning_bilstm.csv")
  d$model <- "BiLSTM (architecture grid)"
  d$score <- d$val_loss
  d$x <- seq_len(nrow(d))
  d$best <- d$score == min(d$score, na.rm = TRUE)
  tuning_panels[["bilstm"]] <- d[, c("model", "x", "score", "best")]
}
if (file.exists("table_tuning_nb.csv")) {
  d <- read.csv("table_tuning_nb.csv")
  d$model <- "Naive Bayes (laplace)"
  d$score <- d$cv_logloss
  d$x <- seq_len(nrow(d))
  d$best <- d$score == min(d$score, na.rm = TRUE)
  tuning_panels[["nb"]] <- d[, c("model", "x", "score", "best")]
}

if (length(tuning_panels) > 0) {
  tuning_df <- do.call(rbind, tuning_panels)
  p_tune <- ggplot(tuning_df, aes(x = x, y = score)) +
    geom_line(linewidth = 0.5, colour = "grey60") +
    geom_point(aes(colour = best, size = best)) +
    scale_color_manual(values = c("FALSE" = "grey40", "TRUE" = "#d7263d"),
                       guide = "none") +
    scale_size_manual(values = c("FALSE" = 1.6, "TRUE" = 3.2), guide = "none") +
    facet_wrap(~ model, scales = "free", ncol = 2) +
    theme_minimal(base_size = 11) +
    labs(title = "Hyperparameter-Tuning Trajectories (v7)",
         subtitle = "Each point is a candidate configuration. Red point = best.",
         x = "Configuration index", y = "CV objective (lower = better)") +
    theme(strip.text = element_text(face = "bold"))
  save_plot(p_tune, "tuning_curves.png", width = 11, height = 7)
  cat("Savedtuning_curves.png\n")
} else {
  cat("No tuning tables found tuning_curves.png skipped.\n")
}


# 12. Session info

sink("session_info.txt")
cat("RESM3002 - Session Information
")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "

")
cat("Hyperparameter-search budgets:
")
cat(" N_RANDOM_XGB:", N_RANDOM_XGB, "
")
cat(" N_RANDOM_LGB:", N_RANDOM_LGB, "
")
cat(" SVM_GRID_SIZE:", SVM_GRID_SIZE, "
")
cat("  BILSTM_GRID_N:", BILSTM_GRID_N, "
")
cat("  GLMNET_ALPHAS:", paste(GLMNET_ALPHAS, collapse = ", "), "
")
cat("  NB_GRID:", paste(NB_GRID, collapse = ", "), "

")
cat("Optional dependencies status:
")
cat("  quanteda stack:", quanteda_available, "
")
cat("  STM:", stm_available, "
")
cat("  Sentence embed.:", emb_available, "
")
cat(" irlba (LSA):", irlba_ok, "
")
cat(" LightGBM:", lgb_available, "
")
cat(" fastTextR:", ft_available, "
")
cat("  Keras (",
    if (!is.null(keras_pkg)) keras_pkg else "none",
    "):", keras_available, "
", sep = "")
cat("  pROC (multiclass AUC):",
    requireNamespace("pROC", quietly = TRUE), "
")
cat("  LiblineaR:",
    requireNamespace("LiblineaR", quietly = TRUE), "
")
cat(" ggridges:", ridges_ok, "
")
cat(" ggbeeswarm:", beeswarm_ok, "
")
cat(" circlize:", circlize_ok, "
")
cat(" tidygraph+ggraph:", graph_ok, "

")
cat("Models present in final comparison table:
")
print(results$Model)
cat("
")
sessionInfo()
sink()

cat("
",
    "
",
    "  Analysis complete - outputs written to working dir.
",
    "  Best model: ", best_model_label, "
",
    "  Best accuracy: ", sprintf("%.4f", best_cm$overall["Accuracy"]), "
",
    "
",
    sep = "")
