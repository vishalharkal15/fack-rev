from api.app import ReviewAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    logger.info("Initializing ReviewAnalyzer...")
    analyzer = ReviewAnalyzer()
    
    test_reviews = [
        # Genuine review
        "This is a genuine review. The product works great and I recommend it.",
        # Suspicious review
        "AMAZING!!! BEST EVER!!! BUY NOW!!! 100% PERFECT!!!",
        # Balanced review
        "Decent product with some minor flaws. Good value overall."
    ]
    
    logger.info("Testing model with sample reviews...")
    for review in test_reviews:
        logger.info(f"\nAnalyzing review: {review}")
        try:
            result = analyzer.analyze_review(review)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error analyzing review: {e}")

if __name__ == "__main__":
    test_model() 