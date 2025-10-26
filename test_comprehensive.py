#!/usr/bin/env python3
"""
Comprehensive accuracy test after neutral sentiment improvements
"""

import sys
sys.path.append('src')
from rag_sentiment_predictor import RAGSentimentPredictor

def comprehensive_accuracy_test():
    """Test overall accuracy with improved neutral detection."""
    
    test_cases = [
        # Clear Positive Cases
        ('Stock surges 20% on strong earnings beat', 'positive'),
        ('Company announces record quarterly profits', 'positive'),
        ('Shares rally on major acquisition deal', 'positive'),
        ('Stock jumps on positive analyst upgrade', 'positive'),
        ('Firm reports exceptional Q3 growth', 'positive'),
        ('Reliance Industries beats revenue estimates', 'positive'),
        
        # Clear Negative Cases
        ('Stock plummets 15% after earnings miss', 'negative'),
        ('Company faces bankruptcy amid losses', 'negative'),
        ('Shares crash on regulatory probe', 'negative'),
        ('Stock tumbles on disappointing results', 'negative'),
        ('Firm warns of significant decline', 'negative'),
        ('TCS faces headwinds from client cuts', 'negative'),
        
        # Neutral Cases (Previously 0% accuracy)
        ('Company reports in-line quarterly results', 'neutral'),
        ('Stock trades flat following mixed earnings', 'neutral'),
        ('Firm maintains guidance for the year', 'neutral'),
        ('Results meet analyst expectations', 'neutral'),
        ('Company announces routine changes', 'neutral'),
        ('RBI policy decision in line with expectations', 'neutral'),
        ('Quarterly performance steady with no surprises', 'neutral'),
        ('Company maintains stable outlook', 'neutral'),
        ('Stock shows minimal movement on sector rotation', 'neutral'),
        ('Firm reports consistent performance', 'neutral'),
        
        # Complex/Edge Cases
        ('Despite strong revenue growth, margins compressed', 'neutral'),
        ('Stock rallies but analysts remain cautious', 'neutral'),
        ('Company beats revenue but misses earnings', 'neutral'),
        ('Nifty opens higher on positive global cues', 'positive'),
        ('FII selling pressure weighs on market', 'negative')
    ]

    predictor = RAGSentimentPredictor()
    correct = 0
    total = len(test_cases)

    # Track by category
    categories = {'positive': [0, 0], 'negative': [0, 0], 'neutral': [0, 0]}

    print('🧪 Comprehensive Accuracy Test (After Neutral Improvements)')
    print('=' * 70)

    for text, expected in test_cases:
        try:
            result = predictor.predict(text, method='rag')
            predicted = result['sentiment'].lower()
            confidence = result['confidence']
            
            categories[expected][1] += 1  # total
            
            is_correct = predicted == expected
            if is_correct:
                correct += 1
                categories[expected][0] += 1  # correct
                
            status = '✓' if is_correct else '✗'
            print(f'{status} {expected:8} → {predicted:8} ({confidence:.1f}%) | {text[:45]}')
            
        except Exception as e:
            print(f'Error: {e}')

    print('=' * 70)
    overall_accuracy = (correct / total) * 100
    print(f'Overall Accuracy: {overall_accuracy:.1f}% ({correct}/{total})')

    print()
    print('📊 Accuracy by Category:')
    for category, (correct_cat, total_cat) in categories.items():
        if total_cat > 0:
            cat_accuracy = (correct_cat / total_cat) * 100
            print(f'  {category.capitalize():8}: {cat_accuracy:.1f}% ({correct_cat}/{total_cat})')

    print()
    print('📈 Improvement Summary:')
    print(f'  Previous Overall: 70.0%')
    print(f'  Current Overall:  {overall_accuracy:.1f}%')
    print(f'  Improvement:      +{overall_accuracy - 70.0:.1f}%')
    
    print()
    print('🎯 Performance Breakdown:')
    best_accuracy = max([correct_cat/total_cat*100 for correct_cat, total_cat in categories.values() if total_cat > 0])
    worst_accuracy = min([correct_cat/total_cat*100 for correct_cat, total_cat in categories.values() if total_cat > 0])
    print(f'  Best Category:  {best_accuracy:.1f}%')
    print(f'  Worst Category: {worst_accuracy:.1f}%')
    print(f'  Consistency:    {best_accuracy - worst_accuracy:.1f}% spread')
    
    return overall_accuracy, categories

if __name__ == "__main__":
    comprehensive_accuracy_test()