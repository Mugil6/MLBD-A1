cat > README.md <<EOF
# Big Data Analysis: Hadoop & Spark Implementation

## Student Details
- **Name:** Mugilan
- **Course:** ML with Big Data

## Project Structure
This repository contains the source code for the Big Data assignment, divided into two frameworks:

### 1. Hadoop MapReduce
- **Directory:** \`src/\`
- **Description:** Java-based MapReduce jobs for text analysis (Questions 4-9).
- **Compilation:** Compiled using Hadoop 3.3.6.

### 2. Apache Spark
- **Directory:** \`spark/\`
- **Description:** PySpark scripts for advanced data analysis (Questions 10-12).
- **Files:**
  - \`q-10_metadata.py\`: Analyzes title lengths and publication years.
  - \`q11_tfidf.py\`: TF-IDF Vectorization and Cosine Similarity.
  - \`q12_network.py\`: Author collaboration network analysis.

## Engineering Note (Question 11)
**Constraint:** The Cosine Similarity algorithm requires a Cartesian Product ($O(N^2)$). On a single-node cluster with limited RAM (<2GB), processing the full 3,000+ book dataset caused inevitable Out-Of-Memory (OOM) errors.

**Optimization:** To demonstrate algorithmic correctness while respecting hardware limits, the execution was performed on a **representative random sample** (Target Book + 99 others). This validated the TF-IDF and Similarity logic without crashing the single-node driver.

## Requirements
- **Hadoop:** v3.3.6
- **Spark:** v3.5.1
- **Python:** v3.x
EOF
