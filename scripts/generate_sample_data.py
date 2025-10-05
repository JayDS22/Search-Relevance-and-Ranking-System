"""
Generate sample documents and queries for testing
"""
import json
import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

# Sample topics and content
TOPICS = {
    'machine_learning': {
        'keywords': ['machine learning', 'neural network', 'deep learning', 'algorithm', 'training', 'model', 'classification', 'regression'],
        'titles': [
            'Introduction to Machine Learning',
            'Deep Learning Fundamentals',
            'Neural Networks Explained',
            'Supervised Learning Algorithms',
            'Unsupervised Learning Methods',
            'Reinforcement Learning Guide',
            'Machine Learning Best Practices',
            'Building ML Models'
        ]
    },
    'data_science': {
        'keywords': ['data science', 'analytics', 'statistics', 'visualization', 'python', 'pandas', 'analysis', 'insights'],
        'titles': [
            'Data Science for Beginners',
            'Statistical Analysis Methods',
            'Data Visualization Techniques',
            'Python for Data Analysis',
            'Pandas Tutorial',
            'Exploratory Data Analysis',
            'Big Data Analytics',
            'Data Mining Strategies'
        ]
    },
    'web_development': {
        'keywords': ['web development', 'javascript', 'html', 'css', 'frontend', 'backend', 'api', 'framework'],
        'titles': [
            'Modern Web Development',
            'JavaScript Essentials',
            'Building REST APIs',
            'Frontend Frameworks Comparison',
            'Responsive Web Design',
            'Web Security Best Practices',
            'Full Stack Development',
            'Web Performance Optimization'
        ]
    },
    'databases': {
        'keywords': ['database', 'sql', 'nosql', 'query', 'indexing', 'optimization', 'mongodb', 'postgresql'],
        'titles': [
            'Database Design Principles',
            'SQL Query Optimization',
            'NoSQL vs SQL Databases',
            'Database Indexing Strategies',
            'MongoDB Tutorial',
            'PostgreSQL Guide',
            'Database Scalability',
            'Data Modeling Best Practices'
        ]
    },
    'cloud_computing': {
        'keywords': ['cloud', 'aws', 'azure', 'gcp', 'serverless', 'kubernetes', 'docker', 'devops'],
        'titles': [
            'Cloud Computing Basics',
            'AWS Services Overview',
            'Kubernetes Guide',
            'Docker Containerization',
            'Serverless Architecture',
            'Cloud Security',
            'DevOps Practices',
            'Multi-Cloud Strategies'
        ]
    }
}


def generate_content(topic_data, length=500):
    """Generate random content based on topic keywords"""
    keywords = topic_data['keywords']
    
    sentences = [
        f"This article covers {random.choice(keywords)} and its applications.",
        f"Understanding {random.choice(keywords)} is crucial for modern development.",
        f"We will explore {random.choice(keywords)} in detail.",
        f"The importance of {random.choice(keywords)} cannot be overstated.",
        f"Learn how {random.choice(keywords)} can improve your workflow.",
        f"Best practices for {random.choice(keywords)} implementation.",
        f"Common challenges with {random.choice(keywords)} and solutions.",
        f"Advanced techniques in {random.choice(keywords)}.",
        f"Real-world applications of {random.choice(keywords)}.",
        f"Getting started with {random.choice(keywords)}."
    ]
    
    # Generate paragraphs
    paragraphs = []
    words_count = 0
    target_words = length
    
    while words_count < target_words:
        paragraph_sentences = random.sample(sentences, k=min(4, len(sentences)))
        paragraph = ' '.join(paragraph_sentences)
        paragraphs.append(paragraph)
        words_count += len(paragraph.split())
    
    return ' '.join(paragraphs)


def generate_documents(num_docs=100):
    """Generate sample documents"""
    documents = []
    
    for i in range(num_docs):
        # Choose random topic
        topic_name = random.choice(list(TOPICS.keys()))
        topic_data = TOPICS[topic_name]
        
        # Choose title
        title = random.choice(topic_data['titles'])
        
        # Generate content
        content_length = random.randint(300, 800)
        content = generate_content(topic_data, content_length)
        
        # Generate metadata
        doc = {
            'id': f'doc_{i+1:04d}',
            'title': title,
            'content': content,
            'metadata': {
                'topic': topic_name,
                'author': f'Author {random.randint(1, 20)}',
                'date': f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
                'views': random.randint(100, 10000),
                'quality_score': round(random.uniform(0.5, 1.0), 2),
                'pagerank': round(random.uniform(0.3, 1.0), 2),
                'freshness': round(random.uniform(0.4, 1.0), 2)
            }
        }
        
        documents.append(doc)
    
    return documents


def generate_queries(num_queries=50):
    """Generate sample queries"""
    queries = []
    
    query_templates = [
        'how to {keyword}',
        'what is {keyword}',
        '{keyword} tutorial',
        '{keyword} best practices',
        'learn {keyword}',
        '{keyword} examples',
        'introduction to {keyword}',
        '{keyword} guide',
        'understanding {keyword}',
        '{keyword} basics'
    ]
    
    for i in range(num_queries):
        topic_name = random.choice(list(TOPICS.keys()))
        topic_data = TOPICS[topic_name]
        keyword = random.choice(topic_data['keywords'])
        template = random.choice(query_templates)
        
        query = template.format(keyword=keyword)
        
        queries.append({
            'id': f'query_{i+1:03d}',
            'query': query,
            'topic': topic_name
        })
    
    return queries


def generate_training_data(documents, num_samples=30):
    """Generate training data with relevance judgments"""
    training_data = []
    
    for i in range(num_samples):
        # Choose a random topic
        topic_name = random.choice(list(TOPICS.keys()))
        topic_data = TOPICS[topic_name]
        keyword = random.choice(topic_data['keywords'])
        
        query = f"learn {keyword}"
        
        # Get documents from same topic (relevant) and other topics (less relevant)
        same_topic_docs = [doc for doc in documents if doc['metadata']['topic'] == topic_name]
        other_topic_docs = [doc for doc in documents if doc['metadata']['topic'] != topic_name]
        
        # Sample documents
        relevant_docs = random.sample(same_topic_docs, min(5, len(same_topic_docs)))
        irrelevant_docs = random.sample(other_topic_docs, min(3, len(other_topic_docs)))
        
        # Assign relevance scores
        doc_ids = []
        relevance_scores = []
        
        for doc in relevant_docs:
            doc_ids.append(doc['id'])
            relevance_scores.append(random.choice([3, 4]))  # Highly relevant
        
        for doc in irrelevant_docs:
            doc_ids.append(doc['id'])
            relevance_scores.append(random.choice([0, 1]))  # Not relevant or barely relevant
        
        training_data.append({
            'query': query,
            'documents': doc_ids,
            'relevance_scores': relevance_scores
        })
    
    return training_data


def main():
    """Generate and save sample data"""
    print("Generating sample data...")
    
    # Create data directory
    config.DATA_DIR.mkdir(exist_ok=True)
    
    # Generate documents
    print("Generating documents...")
    documents = generate_documents(num_docs=100)
    
    with open(config.DOCUMENTS_PATH, 'w') as f:
        json.dump(documents, f, indent=2)
    print(f"Generated {len(documents)} documents -> {config.DOCUMENTS_PATH}")
    
    # Generate queries
    print("Generating queries...")
    queries = generate_queries(num_queries=50)
    
    with open(config.QUERIES_PATH, 'w') as f:
        json.dump(queries, f, indent=2)
    print(f"Generated {len(queries)} queries -> {config.QUERIES_PATH}")
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_training_data(documents, num_samples=30)
    
    with open(config.TRAINING_DATA_PATH, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Generated {len(training_data)} training samples -> {config.TRAINING_DATA_PATH}")
    
    print("\nSample data generation complete!")
    print(f"Total documents: {len(documents)}")
    print(f"Total queries: {len(queries)}")
    print(f"Training samples: {len(training_data)}")


if __name__ == '__main__':
    main()
