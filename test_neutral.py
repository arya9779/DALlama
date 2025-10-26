#!/usr/bin/env python3
"""
Test and improve neutral sentiment detection
"""

import sys
sys.path.append('src')
from rag_sentiment_predictor import RAGSentimentPredictor

def analyze_neutral_issues():
    """Analyze why neutral sentiment detection fails."""
    
    neutral_cases = [
        'Company reports in-line quarterly results',
        'Stock trades flat following mixed earnings', 
        'Firm maintains guidance for the year',
        'Results meet analyst expectations',
        'Company announces routine changes',
        'Stock shows minimal movement on sector rotation',
        'Market sentiment remains mixed amid uncertainty',
        'RBI policy decision in line with expectations',
        'Quarterly performance steady with no surprises',
        'Company maintains stable outlook for next quarter'
    ]

    predictor = RAGSentimentPredictor()

    print('Analyzing Neutral Sentiment Issues:')
    print('=' * 60)

    for text in neutral_cases:
        result = predictor.predict(text, method='rag')
        base_result = predictor.predict_base_model(text)
        
        print(f'Text: {text[:40]}...')
        print(f'  RAG:  {result["sentiment"]:8} ({result["confidence"]:.1f}%)')
        print(f'  Base: {base_result["sentiment"]:8} ({base_result["confidence"]:.1f}%)')
        
        # Check what keywords are being detected
        text_lower = text.lower()
        negative_words = ['fall', 'drop', 'decline', 'weak', 'poor', 'concern', 'worry', 'mixed', 'uncertainty']
        detected_neg = [word for word in negative_words if word in text_lower]
        if detected_neg:
            print(f'  Detected negative keywords: {detected_neg}')
        print()

if __name__ == "__main__":
    analyze_neutral_issues()