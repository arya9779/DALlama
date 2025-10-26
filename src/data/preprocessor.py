"""Text preprocessing and feature engineering for Indian stock market text."""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import spacy
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from loguru import logger

from ..config import config

@dataclass
class ProcessedText:
    """Structure for processed text data."""
    original_text: str
    cleaned_text: str
    tokens: List[str]
    entities: List[Dict]
    stock_symbols: List[str]
    features: Dict
    metadata: Dict

class IndianStockTextPreprocessor:
    """Specialized preprocessor for Indian stock market text."""
    
    def __init__(self):
        # Load spaCy model with automatic download fallback
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Attempting to download...")
            try:
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model downloaded and loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to download spaCy model: {e}. Continuing without spaCy.")
                self.nlp = None
            
        # Indian stock market specific patterns
        self.stock_patterns = self._load_stock_patterns()
        self.financial_lexicon = self._load_financial_lexicon()
        
        # Preprocessing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.number_pattern = re.compile(r'\\b\\d+(?:,\\d{3})*(?:\\.\\d+)?\\b')
        
    def _load_stock_patterns(self) -> Dict[str, List[str]]:
        """Load Indian stock symbol patterns and company name mappings."""
        
        # Common Indian stock symbols and their variations
        stock_patterns = {
            "RELIANCE": ["reliance", "ril", "reliance industries", "mukesh ambani"],
            "TCS": ["tcs", "tata consultancy", "tata consultancy services"],
            "HDFCBANK": ["hdfc bank", "hdfc", "housing development finance"],
            "INFY": ["infosys", "infy", "narayana murthy"],
            "HINDUNILVR": ["hindustan unilever", "hul", "unilever"],
            "ICICIBANK": ["icici bank", "icici"],
            "KOTAKBANK": ["kotak mahindra", "kotak bank", "uday kotak"],
            "LT": ["larsen toubro", "l&t", "larsen & toubro"],
            "ITC": ["itc limited", "itc", "cigarettes"],
            "SBIN": ["sbi", "state bank", "state bank of india"],
            "BHARTIARTL": ["bharti airtel", "airtel", "sunil mittal"],
            "ASIANPAINT": ["asian paints", "asian paint"],
            "MARUTI": ["maruti suzuki", "maruti", "suzuki"],
            "AXISBANK": ["axis bank", "axis"],
            "NESTLEIND": ["nestle india", "nestle", "maggi"]
        }
        
        return stock_patterns
    
    def _load_financial_lexicon(self) -> Dict[str, List[str]]:
        """Load financial sentiment lexicon for Indian markets."""
        
        financial_lexicon = {
            "positive": [
                "bullish", "rally", "surge", "gain", "profit", "growth", "strong",
                "outperform", "beat", "exceed", "upgrade", "buy", "accumulate",
                "momentum", "breakout", "uptrend", "recovery", "expansion",
                "dividend", "bonus", "split", "acquisition", "merger"
            ],
            "negative": [
                "bearish", "crash", "fall", "loss", "decline", "weak", "poor",
                "underperform", "miss", "downgrade", "sell", "avoid", "correction",
                "breakdown", "downtrend", "recession", "slowdown", "debt",
                "bankruptcy", "fraud", "scandal", "investigation", "penalty"
            ],
            "neutral": [
                "stable", "flat", "sideways", "consolidation", "range-bound",
                "maintain", "hold", "unchanged", "steady", "consistent"
            ]
        }
        
        return financial_lexicon
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' [URL] ', text)
        
        # Handle mentions and hashtags
        text = self.mention_pattern.sub(' [MENTION] ', text)
        text = self.hashtag_pattern.sub(' [HASHTAG] ', text)
        
        # Normalize numbers (preserve important financial numbers)
        text = re.sub(r'\\b\\d+(?:,\\d{3})*(?:\\.\\d+)?\\s*(?:crore|lakh|billion|million|thousand)\\b', 
                     ' [AMOUNT] ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        text = text.strip()
        
        # Handle common Indian English patterns
        text = self._normalize_indian_english(text)
        
        return text
    
    def _normalize_indian_english(self, text: str) -> str:
        """Normalize Indian English patterns and financial terms."""
        
        # Common Indian financial terms
        replacements = {
            "crores": "crore",
            "lakhs": "lakh", 
            "nifty50": "nifty 50",
            "sensex": "bse sensex",
            "q1": "quarter 1",
            "q2": "quarter 2", 
            "q3": "quarter 3",
            "q4": "quarter 4",
            "fy": "financial year",
            "yoy": "year over year",
            "qoq": "quarter over quarter",
            "ebitda": "earnings before interest tax depreciation amortization",
            "pat": "profit after tax",
            "pbt": "profit before tax"
        }
        
        for old, new in replacements.items():
            text = re.sub(f'\\b{old}\\b', new, text)
            
        return text
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities and financial entities."""
        
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Extract stock symbols and company names
        stock_entities = self._extract_stock_entities(text)
        entities.extend(stock_entities)
        
        return entities
    
    def _extract_stock_entities(self, text: str) -> List[Dict]:
        """Extract stock symbols and company mentions."""
        
        entities = []
        text_lower = text.lower()
        
        for symbol, patterns in self.stock_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_lower.find(pattern, start)
                        if pos == -1:
                            break
                            
                        entities.append({
                            "text": text[pos:pos+len(pattern)],
                            "label": "STOCK_SYMBOL",
                            "symbol": symbol,
                            "start": pos,
                            "end": pos + len(pattern)
                        })
                        
                        start = pos + 1
        
        return entities
    
    def extract_features(self, text: str, metadata: Dict = None) -> Dict:
        """Extract various features from text."""
        
        features = {}
        
        # Basic text features
        features["text_length"] = len(text)
        features["word_count"] = len(text.split())
        features["sentence_count"] = len(re.split(r'[.!?]+', text))
        
        # Financial sentiment features
        sentiment_scores = self._calculate_sentiment_scores(text)
        features.update(sentiment_scores)
        
        # Market timing features
        if metadata and "timestamp" in metadata:
            timing_features = self._extract_timing_features(metadata["timestamp"])
            features.update(timing_features)
        
        # Source features
        if metadata and "source" in metadata:
            features["source_type"] = self._categorize_source(metadata["source"])
        
        return features
    
    def _calculate_sentiment_scores(self, text: str) -> Dict:
        """Calculate sentiment scores using financial lexicon."""
        
        words = text.lower().split()
        word_set = set(words)
        
        scores = {}
        
        for sentiment, lexicon_words in self.financial_lexicon.items():
            # Count matches
            matches = len(word_set.intersection(set(lexicon_words)))
            scores[f"{sentiment}_word_count"] = matches
            scores[f"{sentiment}_word_ratio"] = matches / len(words) if words else 0
        
        # Overall polarity score
        pos_score = scores["positive_word_ratio"]
        neg_score = scores["negative_word_ratio"]
        scores["polarity_score"] = pos_score - neg_score
        
        return scores
    
    def _extract_timing_features(self, timestamp: datetime) -> Dict:
        """Extract time-based features."""
        
        features = {}
        
        # Market timing
        features["hour"] = timestamp.hour
        features["day_of_week"] = timestamp.weekday()
        features["is_weekend"] = timestamp.weekday() >= 5
        
        # Market hours (IST: 9:15 AM - 3:30 PM)
        market_start = 9.25  # 9:15 AM
        market_end = 15.5    # 3:30 PM
        
        hour_decimal = timestamp.hour + timestamp.minute / 60
        features["is_market_hours"] = market_start <= hour_decimal <= market_end
        features["is_pre_market"] = hour_decimal < market_start
        features["is_post_market"] = hour_decimal > market_end
        
        return features
    
    def _categorize_source(self, source: str) -> str:
        """Categorize the source type."""
        
        source_lower = source.lower()
        
        if any(news_source in source_lower for news_source in ["economic", "money", "mint", "business"]):
            return "financial_news"
        elif "twitter" in source_lower:
            return "social_media"
        elif "reddit" in source_lower:
            return "forum"
        elif any(filing in source_lower for filing in ["bse", "nse", "sebi"]):
            return "regulatory_filing"
        else:
            return "other"
    
    def process_text(self, text: str, metadata: Dict = None) -> ProcessedText:
        """Process a single text with all preprocessing steps."""
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = cleaned_text.split()
        
        # Extract entities
        entities = self.extract_entities(cleaned_text)
        
        # Extract stock symbols
        stock_symbols = list(set([
            entity["symbol"] for entity in entities 
            if entity.get("label") == "STOCK_SYMBOL"
        ]))
        
        # Extract features
        features = self.extract_features(cleaned_text, metadata)
        
        return ProcessedText(
            original_text=text,
            cleaned_text=cleaned_text,
            tokens=tokens,
            entities=entities,
            stock_symbols=stock_symbols,
            features=features,
            metadata=metadata or {}
        )

class WeakSupervisionLabeler:
    """Generate weak labels using heuristics and price movements."""
    
    def __init__(self, price_data: pd.DataFrame = None):
        self.price_data = price_data
        self.preprocessor = IndianStockTextPreprocessor()
    
    def generate_weak_labels(self, texts: List[str], 
                           timestamps: List[datetime],
                           stock_symbols: List[List[str]]) -> List[str]:
        """Generate weak labels using multiple heuristics."""
        
        labels = []
        
        for text, timestamp, symbols in zip(texts, timestamps, stock_symbols):
            # Process text
            processed = self.preprocessor.process_text(text, {"timestamp": timestamp})
            
            # Combine multiple labeling strategies
            lexicon_label = self._lexicon_based_label(processed)
            price_label = self._price_movement_label(symbols, timestamp)
            
            # Combine labels (lexicon takes priority, price as fallback)
            final_label = lexicon_label if lexicon_label != "neutral" else price_label
            labels.append(final_label)
        
        return labels
    
    def _lexicon_based_label(self, processed_text: ProcessedText) -> str:
        """Label based on financial lexicon."""
        
        features = processed_text.features
        
        pos_ratio = features.get("positive_word_ratio", 0)
        neg_ratio = features.get("negative_word_ratio", 0)
        
        # Thresholds for classification
        if pos_ratio > 0.02 and pos_ratio > neg_ratio * 1.5:
            return "positive"
        elif neg_ratio > 0.02 and neg_ratio > pos_ratio * 1.5:
            return "negative"
        else:
            return "neutral"
    
    def _price_movement_label(self, stock_symbols: List[str], 
                            timestamp: datetime) -> str:
        """Label based on price movement after news."""
        
        if not self.price_data or not stock_symbols:
            return "neutral"
        
        # Look at price movement 1-3 days after news
        start_date = timestamp.date()
        end_date = start_date + timedelta(days=3)
        
        movements = []
        
        for symbol in stock_symbols:
            symbol_data = self.price_data[
                (self.price_data["Symbol"] == symbol) &
                (self.price_data["Date"] >= start_date) &
                (self.price_data["Date"] <= end_date)
            ].sort_values("Date")
            
            if len(symbol_data) >= 2:
                # Calculate return
                start_price = symbol_data.iloc[0]["Close"]
                end_price = symbol_data.iloc[-1]["Close"]
                return_pct = (end_price - start_price) / start_price
                movements.append(return_pct)
        
        if movements:
            avg_movement = np.mean(movements)
            
            # Thresholds for labeling
            if avg_movement > 0.02:  # > 2% gain
                return "positive"
            elif avg_movement < -0.02:  # > 2% loss
                return "negative"
        
        return "neutral"

class DatasetBuilder:
    """Build training datasets with proper splits."""
    
    def __init__(self):
        self.preprocessor = IndianStockTextPreprocessor()
    
    def build_dataset(self, raw_data_files: List[Path], 
                     price_data: pd.DataFrame = None) -> pd.DataFrame:
        """Build complete dataset from raw data files."""
        
        all_texts = []
        
        # Process all data files
        for file_path in raw_data_files:
            logger.info(f"Processing {file_path}")
            
            if file_path.suffix == ".json":
                texts = self._process_json_file(file_path)
                all_texts.extend(texts)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_texts)
        
        if df.empty:
            logger.warning("No data found in input files")
            return df
        
        # Generate weak labels
        if price_data is not None:
            labeler = WeakSupervisionLabeler(price_data)
            df["weak_label"] = labeler.generate_weak_labels(
                df["text"].tolist(),
                df["timestamp"].tolist(), 
                df["stock_symbols"].tolist()
            )
        
        # Add processed features
        df = self._add_processed_features(df)
        
        logger.info(f"Built dataset with {len(df)} samples")
        return df
    
    def _process_json_file(self, file_path: Path) -> List[Dict]:
        """Process a JSON data file."""
        
        texts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # Handle different data formats
            if "title" in item and "content" in item:
                # News article
                text = f"{item['title']} {item['content']}"
                timestamp = datetime.fromisoformat(item["published_at"].replace('Z', '+00:00'))
                source = item.get("source", "unknown")
                stock_symbols = item.get("stock_symbols", [])
                
            elif "text" in item:
                # Social media post
                text = item["text"]
                timestamp = item.get("created_at", datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                source = item.get("source", "social")
                stock_symbols = item.get("stock_symbols", [])
                
            else:
                continue
            
            # Filter by text length
            if len(text) < config.data.min_text_length:
                continue
            if len(text) > config.data.max_text_length:
                text = text[:config.data.max_text_length]
            
            texts.append({
                "text": text,
                "timestamp": timestamp,
                "source": source,
                "stock_symbols": stock_symbols
            })
        
        return texts
    
    def _add_processed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add processed text features to dataset."""
        
        processed_features = []
        
        for _, row in df.iterrows():
            processed = self.preprocessor.process_text(
                row["text"], 
                {"timestamp": row["timestamp"], "source": row["source"]}
            )
            
            features = {
                "cleaned_text": processed.cleaned_text,
                "word_count": processed.features["word_count"],
                "polarity_score": processed.features["polarity_score"],
                "source_type": processed.features.get("source_type", "other")
            }
            
            processed_features.append(features)
        
        # Add features to dataframe
        feature_df = pd.DataFrame(processed_features)
        df = pd.concat([df, feature_df], axis=1)
        
        return df
    
    def create_splits(self, df: pd.DataFrame, 
                     time_aware: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits."""
        
        if time_aware:
            # Time-aware split (older data for training)
            df_sorted = df.sort_values("timestamp")
            
            n = len(df_sorted)
            train_end = int(n * config.data.train_split)
            val_end = int(n * (config.data.train_split + config.data.val_split))
            
            train_df = df_sorted.iloc[:train_end]
            val_df = df_sorted.iloc[train_end:val_end]
            test_df = df_sorted.iloc[val_end:]
            
        else:
            # Random split
            train_df, temp_df = train_test_split(
                df, test_size=(1 - config.data.train_split), 
                random_state=42, stratify=df.get("weak_label")
            )
            
            val_size = config.data.val_split / (config.data.val_split + config.data.test_split)
            val_df, test_df = train_test_split(
                temp_df, test_size=(1 - val_size),
                random_state=42, stratify=temp_df.get("weak_label")
            )
        
        logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df

# CLI interface
def main():
    """Main function for preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess stock market text data")
    parser.add_argument("--input-dir", type=Path, default="data/raw", 
                       help="Input directory with raw data files")
    parser.add_argument("--output-dir", type=Path, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--price-data", type=Path, help="Price data CSV file")
    
    args = parser.parse_args()
    
    # Find all JSON files in input directory
    json_files = list(args.input_dir.glob("*.json"))
    
    if not json_files:
        logger.error(f"No JSON files found in {args.input_dir}")
        return
    
    # Load price data if provided
    price_data = None
    if args.price_data and args.price_data.exists():
        price_data = pd.read_csv(args.price_data)
        price_data["Date"] = pd.to_datetime(price_data["Date"])
    
    # Build dataset
    builder = DatasetBuilder()
    df = builder.build_dataset(json_files, price_data)
    
    if df.empty:
        logger.error("No data processed")
        return
    
    # Create splits
    train_df, val_df, test_df = builder.create_splits(df)
    
    # Save processed data
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(args.output_dir / "train.csv", index=False)
    val_df.to_csv(args.output_dir / "val.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)
    
    # Save full dataset
    df.to_csv(args.output_dir / "full_dataset.csv", index=False)
    
    logger.info(f"Processed data saved to {args.output_dir}")

if __name__ == "__main__":
    main()