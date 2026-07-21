import torch
from transformers import pipeline

def summarize_text(article_text: str, max_len: int = 130, min_len: int = 30) -> str:
    """
    Summarizes lengthy input text using a pre-trained BART CNN model.
    """
    # Initialize transformer summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Generate summary
    summary = summarizer(article_text, max_length=max_len, min_length=min_len, do_sample=False)
    
    return summary[0]['summary_text']

if __name__ == "__main__":
    sample_article = """
    Artificial Intelligence (AI) is transforming industries across the global economy by automating complex tasks,
    enhancing analytical decision-making, and fostering new modalities of innovation. Modern deep learning architectures,
    particularly transformer models, have catalyzed breakthroughs in natural language processing, computer vision, and speech recognition.
    Organizations are increasingly integrating these technologies into business strategies to improve operational efficiency,
    tailor customer interactions, and discover hidden trends within massive datasets. However, ethical considerations regarding data privacy,
    algorithmic bias, and workforce disruption remain pivotal focal points in the sustainable deployment of AI platforms.
    """
    
    print("=" * 60)
    print("ORIGINAL TEXT:")
    print("=" * 60)
    print(sample_article.strip())
    
    print("\n" + "=" * 60)
    print("GENERATED SUMMARY:")
    print("=" * 60)
    summary_output = summarize_text(sample_article)
    print(summary_output)