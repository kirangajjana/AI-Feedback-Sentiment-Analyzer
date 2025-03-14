import os
from phi.agent import Agent
from phi.model.openai import OpenAIChat
# from phi.model.google import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import re

# Load environment variables
load_dotenv()

# Initialize LLM model
gpt_model = OpenAIChat(id="gpt-4o")  # GPT-4o for classification & positive feedback

# Create Sentiment Analysis Agent with clearer output format instructions
sentiment_agent = Agent(
    name="SentimentClassifier",
    role="Classifies feedback as 'positive', 'negative', or 'neutral'. Returns confidence score.",
    model=gpt_model,
    instructions=[
        "Classify the following feedback sentiment as either 'positive', 'negative', or 'neutral'.",
        "Return ONLY the classification word followed by comma and then the confidence score between 0 and 1.",
        "Format your response exactly like this: 'negative, 0.95' or 'positive, 0.85' or 'neutral, 0.60'.",
        "Use decimal points for confidence scores, not percentages. No explanation needed."
    ],
)

# Create Positive Feedback Response Agent
positive_agent = Agent(
    name="PositiveFeedbackResponder",
    role="Generates an engaging response for positive feedback.",
    model=gpt_model,
    instructions=[
        "Write a polite and professional response to this positive feedback.",
        "Make the user feel appreciated and valued.",
        "Keep the response short and engaging."
    ],
)

# Create Negative Feedback Response Agent
negative_agent = Agent(
    name="NegativeFeedbackResponder",
    role="Generates an empathetic response for negative feedback and suggests improvements.",
    model=gpt_model,
    instructions=[
        "Write a professional, empathetic response to this negative feedback.",
        "Acknowledge the issue and suggest a possible improvement or resolution.",
        "Ensure a polite and reassuring tone."
    ],
)

# Create Neutral Feedback Response Agent
neutral_agent = Agent(
    name="NeutralFeedbackResponder",
    role="Generates a balanced response for neutral feedback.",
    model=gpt_model,
    instructions=[
        "Write a professional response to this neutral feedback.",
        "Thank the user for their input and encourage more detailed feedback if possible.",
        "Keep the response balanced and forward-looking."
    ],
)

# Helper function to get content from RunResponse
def get_response_content(response):
    """Extract content from a RunResponse object"""
    if hasattr(response, 'content'):
        return response.content
    return str(response)

# Improved parse sentiment response function
def parse_sentiment(response):
    # Extract the content from the RunResponse object
    response_text = get_response_content(response).strip().lower()
    
    # Debug the response (for development)
    print(f"Raw sentiment response: {response_text}")
    
    # Direct pattern match for the expected format "sentiment, score"
    pattern = r"(positive|negative|neutral),\s*([0-9]*\.?[0-9]+)"
    match = re.search(pattern, response_text)
    
    if match:
        sentiment = match.group(1)
        confidence = float(match.group(2))
        return sentiment, confidence
    
    # Fallback parsing if the pattern doesn't match
    # Check for sentiment keywords
    if "negative" in response_text:
        sentiment = "negative"
    elif "positive" in response_text:
        sentiment = "positive"
    elif "neutral" in response_text:
        sentiment = "neutral"
    else:
        # Default to neutral if no sentiment is found
        sentiment = "neutral"
    
    # Try to extract confidence score
    numbers = re.findall(r"[0-9]*\.?[0-9]+", response_text)
    confidence = float(numbers[0]) if numbers else 0.5
    
    return sentiment, confidence

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Feedback Sentiment Analyzer",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("AI Feedback Sentiment Analyzer")
    st.subheader("Analyze customer feedback and generate appropriate responses")
    
    with st.container():
        st.markdown("### Enter Customer Feedback")
        feedback = st.text_area("Type or paste customer feedback here:", height=150)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            analyze_button = st.button("Analyze Feedback", type="primary")
        with col2:
            if st.button("Clear", type="secondary"):
                st.session_state.clear()
                st.rerun()
    
    if analyze_button and feedback:
        with st.spinner("Analyzing sentiment..."):
            # Get sentiment classification
            sentiment_response = sentiment_agent.run(feedback)
            sentiment, confidence = parse_sentiment(sentiment_response)
            
            # Store raw response for debugging
            raw_response = get_response_content(sentiment_response)
            st.session_state.debug_message = f"Raw response: {raw_response}\nParsed: {sentiment}, {confidence}"
        
        # Display sentiment analysis results
        st.markdown("### Analysis Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment.capitalize())
        with col2:
            st.metric("Confidence", f"{confidence:.2f}")
        
        # Color coding for sentiment
        sentiment_colors = {
            "positive": "green",
            "negative": "red",
            "neutral": "blue"
        }
        
        st.markdown(f"<div style='padding:10px; background-color:{sentiment_colors[sentiment]}20; border-left:5px solid {sentiment_colors[sentiment]}; margin-bottom:20px;'>Sentiment: <strong>{sentiment.capitalize()}</strong> with confidence score of <strong>{confidence:.2f}</strong></div>", unsafe_allow_html=True)
        
        # Generate appropriate response based on sentiment
        st.markdown("### Generated Response")
        
        with st.spinner("Generating response..."):
            if sentiment == "positive":
                response = positive_agent.run(feedback)
                response_text = get_response_content(response)
            elif sentiment == "negative":
                response = negative_agent.run(feedback)
                response_text = get_response_content(response)
            else:  # neutral
                response = neutral_agent.run(feedback)
                response_text = get_response_content(response)
                
        st.text_area("Response:", value=response_text, height=150)
        
        # Add copy button for the response
        if st.button("Copy Response to Clipboard"):
            st.toast("Response copied to clipboard!")
        
        # Feedback metrics
        st.markdown("### Recent Feedback Statistics")
        
        # Initialize or update session state for metrics
        if 'total_feedbacks' not in st.session_state:
            st.session_state.total_feedbacks = 0
            st.session_state.positive_count = 0
            st.session_state.negative_count = 0
            st.session_state.neutral_count = 0
        
        st.session_state.total_feedbacks += 1
        if sentiment == "positive":
            st.session_state.positive_count += 1
        elif sentiment == "negative":
            st.session_state.negative_count += 1
        else:
            st.session_state.neutral_count += 1
        
        # Display metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Feedbacks", st.session_state.total_feedbacks)
        with metric_cols[1]:
            positive_pct = (st.session_state.positive_count / st.session_state.total_feedbacks) * 100
            st.metric("Positive", f"{positive_pct:.1f}%")
        with metric_cols[2]:
            negative_pct = (st.session_state.negative_count / st.session_state.total_feedbacks) * 100
            st.metric("Negative", f"{negative_pct:.1f}%")
        with metric_cols[3]:
            neutral_pct = (st.session_state.neutral_count / st.session_state.total_feedbacks) * 100
            st.metric("Neutral", f"{neutral_pct:.1f}%")
        
        # Add debug expander (can be removed in production)
        with st.expander("Debug Info", expanded=False):
            st.text(st.session_state.debug_message if "debug_message" in st.session_state else "No debug info")
    
    # Add helpful instructions at the bottom
    with st.expander("How to use this app"):
        st.markdown("""
        1. Enter customer feedback in the text area above
        2. Click 'Analyze Feedback' to process the input
        3. Review the sentiment analysis and confidence score
        4. Use the generated response for your customer communications
        5. The app will track feedback statistics across your session
        """)

if __name__ == "__main__":
    main()