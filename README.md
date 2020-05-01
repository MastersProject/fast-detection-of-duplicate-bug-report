# Fast Detection of Duplicate Bug Report

A bug tracking system continuously monitors the status of a software environment, like an Operating System (OS) or user applications. Whenever it detects an anomaly situation, it generates a bug report and sends it out to the software developer or maintenance center. However, the newly reported bug can be an already existing issue that was reported earlier and may have a solution in the master report repository at the developer side. Such instances may occur repeatedly in an overwhelming number. This poses a big challenge to the developer. Thus, early detection of duplicate bug reports has become an extremely important task. This work proposes a double-tier approach using clustering and classification, whereby it exploits Latent Dirichlet Allocation (LDA) for topic-based clustering, single and multimodal text representation using Word2Vec (W2V), FastText (FT), and Global Vectors for Word Representation (GloVe), and text similarity measure fusing Cosine and Euclidean measures. The proposed model is tested on the Eclipse dataset consisting over 80,000 bug reports, which is the amalgamation of both master and duplicate reports. This work only considers the description of the reports for detecting duplicates. The experimental results show that the proposed double-tier model achieves a recall rate of 67% for Top-N recommendations in 3 times faster computation than the conventional one-on-one classification model. 

# Flow of the Proposed Double-tier Approach
![](/flow_diagram.png)

## Stage 1 : Clustering
  - Latent Dirichlet Allocation (LDA) is applied on the preprocessed master reports toform clusters.
  - Pre-trained LDA is applied on preproccesed duplicate report to find the most similar cluster in which associated master report may         exist.
## Stage 2 : Classification 
  - Home cluster : The duplicate report jumps into the selected cluster to find the most similar master report.
  - Finding the master report:
      * Unified similarity measure using Cosine and Euclidean similarity is used to find the similarity measure between single or                 multimodal feature of the duplicate report and the master reports in the corresponding cluster individually.
      * Top-N similarities would be selected which would result in Top-N recommended master reports.
