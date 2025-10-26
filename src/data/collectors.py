"""Data collection modules for various sources."""

import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import json
import time
from loguru import logger

from ..config import config

@dataclass
class NewsArticle:
    """Structure for news articles."""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    stock_symbols: List[str]
    author: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "stock_symbols": self.stock_symbols,
            "author": self.author
        }

class NewsCollector:
    """Collect news articles from various Indian financial news sources."""
    
    def __init__(self):
        self.session = None
        self.rate_limit_delay = 1.0  # seconds between requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_economic_times(self, stock_symbols: List[str], 
                                   days_back: int = 30) -> List[NewsArticle]:
        """Collect articles from Economic Times."""
        articles = []
        
        for symbol in stock_symbols:
            try:
                # ET RSS feeds or search API
                search_url = f"https://economictimes.indiatimes.com/markets/stocks/news"
                
                # Simulate API call (replace with actual ET API)
                await asyncio.sleep(self.rate_limit_delay)
                
                # Mock data for demonstration
                mock_article = NewsArticle(
                    id=f"et_{symbol}_{int(time.time())}",
                    title=f"Market Analysis: {symbol} Shows Strong Performance",
                    content=f"Detailed analysis of {symbol} stock performance...",
                    source="Economic Times",
                    url=f"https://economictimes.indiatimes.com/markets/{symbol.lower()}",
                    published_at=datetime.now() - timedelta(days=1),
                    stock_symbols=[symbol]
                )
                articles.append(mock_article)
                
                logger.info(f"Collected ET article for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting ET data for {symbol}: {e}")
                
        return articles
    
    async def collect_moneycontrol(self, stock_symbols: List[str], 
                                 days_back: int = 30) -> List[NewsArticle]:
        """Collect articles from Moneycontrol."""
        articles = []
        
        for symbol in stock_symbols:
            try:
                # Moneycontrol API endpoint
                await asyncio.sleep(self.rate_limit_delay)
                
                # Mock data
                mock_article = NewsArticle(
                    id=f"mc_{symbol}_{int(time.time())}",
                    title=f"{symbol}: Quarterly Results Beat Expectations",
                    content=f"Comprehensive coverage of {symbol} quarterly performance...",
                    source="Moneycontrol",
                    url=f"https://moneycontrol.com/news/{symbol.lower()}",
                    published_at=datetime.now() - timedelta(hours=6),
                    stock_symbols=[symbol]
                )
                articles.append(mock_article)
                
                logger.info(f"Collected Moneycontrol article for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting Moneycontrol data for {symbol}: {e}")
                
        return articles

class SocialMediaCollector:
    """Collect social media posts about stocks."""
    
    def __init__(self, twitter_bearer_token: Optional[str] = None):
        self.twitter_token = twitter_bearer_token or config.twitter_bearer_token
        
    async def collect_twitter_posts(self, stock_symbols: List[str], 
                                  days_back: int = 7) -> List[Dict]:
        """Collect Twitter posts mentioning stock symbols."""
        posts = []
        
        if not self.twitter_token:
            logger.warning("Twitter bearer token not provided")
            return posts
            
        for symbol in stock_symbols:
            try:
                # Twitter API v2 search
                query = f"${symbol} OR {symbol} lang:en -is:retweet"
                
                # Mock Twitter data
                mock_post = {
                    "id": f"tw_{symbol}_{int(time.time())}",
                    "text": f"Bullish on ${symbol}! Great fundamentals and strong growth prospects.",
                    "author_id": "mock_user_123",
                    "created_at": datetime.now() - timedelta(hours=2),
                    "public_metrics": {"like_count": 15, "retweet_count": 3},
                    "stock_symbols": [symbol],
                    "source": "Twitter"
                }
                posts.append(mock_post)
                
                logger.info(f"Collected Twitter posts for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting Twitter data for {symbol}: {e}")
                
        return posts
    
    async def collect_reddit_posts(self, stock_symbols: List[str]) -> List[Dict]:
        """Collect Reddit posts from investing subreddits."""
        posts = []
        
        subreddits = ["IndiaInvestments", "SecurityAnalysis", "ValueInvesting"]
        
        for symbol in stock_symbols:
            try:
                # Reddit API search
                # Mock Reddit data
                mock_post = {
                    "id": f"rd_{symbol}_{int(time.time())}",
                    "title": f"DD: Why {symbol} is undervalued",
                    "selftext": f"Detailed analysis of {symbol} showing strong value proposition...",
                    "subreddit": "IndiaInvestments",
                    "created_utc": datetime.now() - timedelta(days=1),
                    "score": 45,
                    "num_comments": 12,
                    "stock_symbols": [symbol],
                    "source": "Reddit"
                }
                posts.append(mock_post)
                
                logger.info(f"Collected Reddit posts for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting Reddit data for {symbol}: {e}")
                
        return posts

class PriceDataCollector:
    """Collect historical price data for backtesting."""
    
    def __init__(self):
        self.cache_dir = Path("data/raw/prices")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_price_data(self, stock_symbols: List[str], 
                          period: str = "1y") -> pd.DataFrame:
        """Collect price data using yfinance."""
        
        # Convert to NSE format
        nse_symbols = [f"{symbol}.NS" for symbol in stock_symbols]
        
        try:
            # Download data
            data = yf.download(nse_symbols, period=period, group_by='ticker')
            
            # Process and clean data
            price_data = []
            
            for symbol in stock_symbols:
                nse_symbol = f"{symbol}.NS"
                if nse_symbol in data.columns.levels[0]:
                    symbol_data = data[nse_symbol].copy()
                    symbol_data['Symbol'] = symbol
                    symbol_data['Date'] = symbol_data.index
                    symbol_data = symbol_data.reset_index(drop=True)
                    price_data.append(symbol_data)
            
            if price_data:
                combined_data = pd.concat(price_data, ignore_index=True)
                
                # Save to cache
                cache_file = self.cache_dir / f"prices_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
                combined_data.to_csv(cache_file, index=False)
                
                logger.info(f"Collected price data for {len(stock_symbols)} symbols")
                return combined_data
            
        except Exception as e:
            logger.error(f"Error collecting price data: {e}")
            
        return pd.DataFrame()

class DataCollectionPipeline:
    """Main pipeline for collecting all data sources."""
    
    def __init__(self):
        self.news_collector = None
        self.social_collector = SocialMediaCollector()
        self.price_collector = PriceDataCollector()
        
    async def collect_all_data(self, stock_symbols: Optional[List[str]] = None,
                             days_back: int = 30) -> Dict[str, List]:
        """Collect data from all sources."""
        
        if stock_symbols is None:
            stock_symbols = config.data.target_stocks
            
        logger.info(f"Starting data collection for {len(stock_symbols)} stocks")
        
        all_data = {
            "news_articles": [],
            "social_posts": [],
            "price_data": None
        }
        
        # Collect news articles
        async with NewsCollector() as news_collector:
            et_articles = await news_collector.collect_economic_times(stock_symbols, days_back)
            mc_articles = await news_collector.collect_moneycontrol(stock_symbols, days_back)
            
            all_data["news_articles"].extend(et_articles)
            all_data["news_articles"].extend(mc_articles)
        
        # Collect social media posts
        twitter_posts = await self.social_collector.collect_twitter_posts(stock_symbols, days_back)
        reddit_posts = await self.social_collector.collect_reddit_posts(stock_symbols)
        
        all_data["social_posts"].extend(twitter_posts)
        all_data["social_posts"].extend(reddit_posts)
        
        # Collect price data
        all_data["price_data"] = self.price_collector.collect_price_data(stock_symbols)
        
        # Save collected data
        await self._save_collected_data(all_data)
        
        logger.info(f"Data collection completed: {len(all_data['news_articles'])} articles, "
                   f"{len(all_data['social_posts'])} social posts")
        
        return all_data
    
    async def _save_collected_data(self, data: Dict) -> None:
        """Save collected data to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save news articles
        if data["news_articles"]:
            news_file = config.data.raw_data_dir / f"news_articles_{timestamp}.json"
            with open(news_file, 'w', encoding='utf-8') as f:
                json.dump([article.to_dict() for article in data["news_articles"]], 
                         f, indent=2, ensure_ascii=False)
        
        # Save social posts
        if data["social_posts"]:
            social_file = config.data.raw_data_dir / f"social_posts_{timestamp}.json"
            with open(social_file, 'w', encoding='utf-8') as f:
                json.dump(data["social_posts"], f, indent=2, ensure_ascii=False, default=str)
        
        # Price data is saved by PriceDataCollector
        
        logger.info(f"Data saved with timestamp {timestamp}")

# CLI interface
async def main():
    """Main function for running data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect stock market data")
    parser.add_argument("--stocks", nargs="+", help="Stock symbols to collect")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    
    args = parser.parse_args()
    
    pipeline = DataCollectionPipeline()
    await pipeline.collect_all_data(args.stocks, args.days)

if __name__ == "__main__":
    asyncio.run(main())