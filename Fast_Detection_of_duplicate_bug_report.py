"""
Detection of Duplicate Bug Report using LDA-based Topic Modeling and Classification Pipeline

This module implements a comprehensive duplicate detection system using:
- Text preprocessing
- Topic modeling with Latent Dirichlet Allocation (LDA)
- Word embedding techniques (Word2Vec and GloVe)
- Multi-modal similarity detection

Key Components:
- TextPreprocessor: Cleans and prepares text data
- TopicModeler: Clusters documents using topic modeling
- DuplicateDetector: Identifies potential duplicate reports
"""

import re
import string
import numpy as np
import pandas as pd
import logging

# NLP and Machine Learning Libraries
import nltk
import gensim
import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec, CoherenceModel
from gensim.test.utils import get_tmpfile
from glove import Glove, Corpus

# Download NLTK resources
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration parameters for the Detection of duplicate bug reports pipeline."""
    # Text Preprocessing
    MIN_TOKEN_LENGTH = 5
    
    # Topic Modeling
    NUM_TOPICS = 10
    LDA_PASSES = 20
    LDA_ITERATIONS = 100
    
    # Word Embeddings
    VECTOR_SIZE = 100
    WINDOW_SIZE = 6
    MIN_WORD_COUNT = 5
    
    # Duplicate Detection
    TOP_N_CANDIDATES = 3
    TOP_N_REPORTS = 833
    TEST_SAMPLES = 200

class TextPreprocessor:
    
    """Handles comprehensive text preprocessing for duplicate detection.
    
    Provides methods for cleaning, lemmatizing, and tokenizing text."""
    
    
    @staticmethod
    def clean_text_round1(text: str) -> str:
        """
        Perform initial text cleaning: remove numbers, brackets, and standardize text.
        
        Args:
            text (str): Input text to clean
        
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove numbers, form feeds, and text within brackets
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\w*\f\w*', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        
        return text

    @staticmethod
    def clean_text_round2(text: str) -> str:
        """
        Perform secondary text cleaning: remove additional punctuation and whitespace.
        
        Args:
            text (str): Input text to clean
        
        Returns:
            str: Further cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove specific punctuation and whitespace
        text = re.sub(r"[''""…]", '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\t', ' ', text)
        text = text.strip()
        
        return text

    @staticmethod
    def lemmatize(text: str) -> str:

        """Lemmatize text using WordNet lemmatizer.
        
        Args:
            text (str): Input text to lemmatize
        
        Returns:
            str: Lemmatized text
        """
        return WordNetLemmatizer().lemmatize(text, pos='v')

    @staticmethod
    def preprocess(text: str) -> list[str]:

        """
        Perform comprehensive text preprocessing.
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            List[str]: Preprocessed tokens
        """
        tokens = gensim.utils.simple_preprocess(text)
        return [
            TextPreprocessor.lemmatize(token) 
            for token in tokens 
            if token not in STOPWORDS and len(token) > Config.MIN_TOKEN_LENGTH
        ]
class TopicModeler:

    '''Performs topic modeling on master reports using Latent Dirichlet Allocation (LDA).
    
    Helps in clustering documents into meaningful topics.'''

    def __init__(self, num_topics: int = Config.NUM_TOPICS):
        """
        Initialize topic modeling parameters.
        
        Args:
            num_topics (int): Number of topics for LDA
            
        """
        self.num_topics = num_topics   
        self.lda_model = None
        self.dictionary = None
        self.topic_clusters: list[pd.DataFrame] = []

    def topic_modeling(self, master_reports: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Create Clusters using LDA based topic modeling .
        
        Args:
            master_reports (pd.DataFrame): DataFrame of master reports

        Returns:
            List[pd.DataFrame]: Clusters of documents by topic

        """
        try:
            # Create dictionary from document descriptions
            self.dictionary = gensim.corpora.Dictionary(master_reports['Description'])
            self.dictionary.filter_extremes(no_below=15, # Ignore words that appear in less than 15 documents
                                            no_above=0.5, # Ignore words that appear in more than 50% of documents 
                                            keep_n=100000  # Keep only top 100,000 terms
                                            )
            
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in master_reports['Description']]

            # Train LDA model
            self.lda_model = gensim.models.LdaMulticore(
                corpus=bow_corpus, 
                num_topics=self.num_topics, 
                id2word=self.dictionary, 
                passes=Config.LDA_PASSES, 
                workers=2, 
                iterations=Config.LDA_ITERATIONS
            )
            
            # create topic Clusters of  master reports by dominant topic
            for c in range(self.num_topics):
                topic_cluster = master_reports[
                    master_reports['Description'].apply(
                        lambda doc: np.argmax(self.lda_model[self.dictionary.doc2bow(doc)]) == c
                    )
                ]
                self.topic_clusters.append(topic_cluster)
            return(self.topic_clusters)

        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            raise

class DuplicateDetector:

    """ 
    Detects duplicate reports using advanced word embedding and similarity techniques.
    
    Combines Word2Vec and GloVe embeddings with a unified similarity metric.
    
     """
    
    def __init__(
        self, 
        topic_clusters: list[pd.DataFrame], # Pass topic clusters from TopicModeler
        vector_size: int = Config.VECTOR_SIZE,
        window_size: int = Config.WINDOW_SIZE,
        min_count: int = Config.MIN_WORD_COUNT

    ):
        """
        Initialize duplicate detection parameters.
        
        Args:
            topic_clusters (list[pd.DataFrame]): Clusters of documents by topic
            vector_size (int): Dimension of word embeddings
            window_size (int): Context window size for word2vec
            min_count (int): Minimum word frequency to be included
        """
        self.topic_clusters = topic_clusters
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        
        self.word2vec_models: list[Word2Vec] = []
        self.glove_models: list[Glove] = []

    
            
    def train_word_embeddings(self) -> None:

        """
        Train Word2Vec and GloVe models for each topic cluster.
        
        Trains embeddings to capture semantic relationships within each cluster.
        
        """

        for cluster_idx, cluster in enumerate(self.topic_clusters):
            try:
                # Prepare corpus for Word2Vec and GloVe
                corpus = cluster['Description'].apply(gensim.utils.simple_preprocess).tolist()

                logger.info(f"Training Word2Vec for cluster {cluster_idx}...")

                # Train Word2Vec
                w2v_model = Word2Vec(
                    corpus, 
                    vector_size=self.vector_size, 
                    window=self.window_size, 
                    min_count=self.min_count, 
                    sg=0,  # CBOW
                    epochs=100
                )
                self.word2vec_models.append(w2v_model)
                
                # Train GloVe
                glove_corpus = Corpus()
                glove_corpus.fit(corpus)

                logger.info(f"Training Glove for cluster {cluster_idx}...")

                glove_model = Glove(
                    no_components=self.vector_size, 
                    learning_rate=0.18, 
                    alpha=0.75, 
                    max_count=100, 
                    max_loss=10.0
                )
                glove_model.fit(glove_corpus.matrix, epochs=1000, no_threads=3)
                glove_model.add_dictionary(glove_corpus.dictionary)
                self.glove_models.append(glove_model)

             except Exception as e:
                logger.error(f"Error training embeddings for cluster {cluster_idx}: {e}")
                raise



    @staticmethod
    def average_word_vectors(words: list[str], model, vocabulary: set[str], num_features: int) -> np.ndarray:
        """
        Convert multiple word embeddings into a single document vector.
        
        Args:
            words (list[str]): List of words
            model: Word embedding model
            vocabulary (set[str]): Vocabulary set
            num_features (int): Embedding dimension
        
        Returns:
            np.ndarray: Averaged document vector
        """
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords += 1.
                feature_vector = np.add(feature_vector, model[word])
        
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        
        return feature_vector

    @staticmethod
    def unified_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity between two vectors using average of cosine and euclidean similarities.
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
        
        Returns:
            float: Unified similarity score
        """
        sim1 = 1 / (1 + distance.euclidean(vec1, vec2))
        sim2 = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        return (sim1 + sim2) / 2

    def _find_candidate_clusters(self, sample: list[str], top_n: int = 3) -> list[int]:
        """
        Find top-N candidate clusters for a given sample using LDA.
        
        Args:
            sample (list[str]): Preprocessed sample document
            top_n (int): Number of top clusters to return
        
        Returns:
            list[int]: Indices of top candidate clusters
        """
        try: 
            vec_bow = self.dictionary.doc2bow(sample)
            topic_distribution = self.lda_model[vec_bow]
            topic_array = np.asarray(topic_distribution)
            
            return list(
                topic_array[np.argsort(topic_array[:, 1])[-top_n:][::-1], 0].astype(int)
            )
        except Exception as e:
            logger.error(f"Error finding candidate clusters: {e}")
            raise

    def _compute_similarities(
        self, 
        sample: list[str], 
        cluster: pd.DataFrame, 
        w2v_model, 
        glove_model,
        pca_model
    ) -> np.ndarray:
        """
        Compute similarities between a sample and cluster documents.
        
        Args:
            sample (list[str]): Preprocessed sample document
            cluster (pd.DataFrame): Cluster of documents
            w2v_model: Word2Vec model
            glove_model: GloVe model
            
        
        Returns:
            np.ndarray: Similarity scores
        """
        w2v_vocab = set(w2v_model.wv.index_to_key)
        glove_vocab = set(glove_model.dictionary.keys())
        
        # Compute document vectors
        w2v_sample_vec = self.average_word_vectors(
            sample, w2v_model.wv, w2v_vocab, self.vector_size
        )
        glove_sample_vec = self.average_word_vectors(
            sample, glove_model.dictionary, glove_vocab, self.vector_size
        )
        
        # Fuse vectors using PCA
        pca_model = PCA(n_components=self.vector_size)
        fused_sample_vec = pca_model.fit_transform(
            np.concatenate([w2v_sample_vec, glove_sample_vec]).reshape(1, -1)
        )[0]
        
        # Compute similarities
        similarities = []
        for doc in cluster['Description']:
            w2v_doc_vec = self.average_word_vectors(
                doc, w2v_model.wv, w2v_vocab, self.vector_size
            )
            glove_doc_vec = self.average_word_vectors(
                doc, glove_model.dictionary, glove_vocab, self.vector_size
            )
            
            fused_doc_vec = pca.transform(
                np.concatenate([w2v_doc_vec, glove_doc_vec]).reshape(1, -1)
            )[0]
            
            similarities.append(
                self.unified_similarity(fused_sample_vec, fused_doc_vec)
            )
        
        return np.array(similarities)

    def detect_duplicates(
        self, 
        test_reports: pd.DataFrame, 
        top_n: int =  Config.TOP_N_REPORTS,  #Top-N where N = n * topn so (2.5K = 3*833)
        test_samples: int = Config.TEST_SAMPLES) -> float:
        """
        Detect duplicate reports using multi-modal feature extraction. Calculating Recall rate for Top 2.5 k reports. 
        
        Args:
            test_reports (pd.DataFrame): DataFrame of test (potential duplicate) reports
            top_n (int): Number of top similar reports to consider
            test_samples (int): Number of test samples to process
        
        Returns:
            float: Recall rate
        """
        try:
            vec_acc = []

            for i in range(test_samples):

                logger.info(f"Processing test sample {i+1}/{test_samples}...")
                sample = test_reports.Description.iloc[i]
                sample = TextPreprocessor.preprocess(sample)

                # Find candidate clusters using LDA topic distribution
                candidate_clusters = self._find_candidate_clusters(sample)
                
                detection_results = []
                for cluster_idx in candidate_clusters:
                    cluster = self.topic_clusters[cluster_idx]
                    w2v_model = self.word2vec_models[cluster_idx]
                    glove_model = self.glove_models[cluster_idx]
        
                    
                    similarities = self._compute_similarities(
                        sample, cluster, w2v_model, glove_model,pca_model
                    )

                    # Get top similar report IDs
                    top_similar_indices = np.argsort(similarities)[-top_n:][::-1]
                    detection_results.extend(
                        cluster.Issue_id.iloc[top_similar_indices].tolist()
                    )
                
                # Check if any detected report matches the ground truth
                vec_acc.append(
                    int(test_reports.Duplicated_issue[i] in detection_results)
                )
            
            # Calculate recall rate
            recall_rate = (sum(vec_acc) / len(vec_acc)) * 100
        
            
            return recall_rate
        except Exception as e:
            logger.error(f"Error in duplicate bug report detection: {e}")
            raise

    

def load_and_preprocess_data(file_path: str) -> tuple:
    """
    Load and preprocess input data for duplicate bug reports detection.
    
    Args:
        filepath (str): Path to input CSV file
    
    Returns:
        tuple: Preprocessed master and duplicate reports
    """
    try:
        # Load data
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)

        # Validate required columns
        required_columns = [
            'Description', 'Title', 'Issue_id', 'Duplicated_issue'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove unnecessary columns
        columns_to_drop = [
            'Priority', 'Component', 'Status', 'Resolution', 
            'Version', 'Created_time', 'Resolved_time'
        ]
    
        # Data cleaning
        data = data.dropna(subset=['Description'])

        # Appy  TextPreprocessing
        preprocessor = TextPreprocessor()
        data['Description'] = data['Description'].apply(preprocessor.clean_text_round1)
        data['Title'] = data['Title'].apply(preprocessor.clean_text_round2)
        data['Description'] = data['Description'].apply(preprocessor.clean_text_round2)
    
        data['Title'] = data['Title'].map(preprocessor.preprocess)
        data['Description'] = data['Description'].map(preprocessor.preprocess)
    
        # Separate master and duplicate reports
        master_reports = data[data['Duplicated_issue'].isnull()]
        duplicate_reports = data.dropna(subset=['Duplicated_issue'])

        # Log preprocessing statistics
        logger.info(f"Total records: {len(data)}")
        logger.info(f"Master reports: {len(master_reports)}")
        logger.info(f"Duplicate reports: {len(duplicate_reports)}")

        return master_reports, duplicate_reports

    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("Input file is empty")
        raise ValueError("Input file contains no data")
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise
        
def main(): 
    """
    Main execution function for duplicate detection pipeline.

    Orchestrates the entire duplicate detection process.
    """
    try:
        # 1. Load and preprocess data
        master_reports, duplicate_reports = load_and_preprocess_data('input_data.csv')
        
        # 2. Initialize Topic Modeling
        topic_modeler = TopicModeler() 

        # 3. Perform Topic Modeling to create topic clusters
        topic_clusters = topic_modeler.topic_modeling(master_reports)

        # 4. Log topic cluster information
        logger.info(f"Number of topic clusters created: {len(topic_clusters)}")
        for i, cluster in enumerate(topic_clusters):
            logger.info(f"Cluster {i}: {len(cluster)} documents")

        # 5. Initialize Duplicate Detector with topic clusters
        duplicate_detector = DuplicateDetector(topic_clusters)

        # 6. Train Word Embeddings for each topic cluster
        duplicate_detector.train_word_embeddings()
        
        # 7. Detect Duplicates
        recall_rate = duplicate_detector.detect_duplicates(
            duplicate_reports,
            top_n=833,       # Number of top similar reports to consider
            test_samples=200 # Number of test samples to process
        )

        # 8. Print final results
        logger.info(f"Duplicate Detection Recall Rate: {recall_rate:.2f}%")
        
        results_df = pd.DataFrame({
            'Recall Rate': [recall_rate],
            'Num Topics': [len(topic_clusters)],
            'Total Master Reports': [len(master_reports)],
            'Total Duplicate Reports': [len(duplicate_reports)]
        })
        results_df.to_csv('duplicate_detection_results.csv', index=False)
    
    except Exception as e:
        logger.error(f"Duplicate detection pipeline failed: {e}")
        raise
        

if __name__ == "__main__":
    main()