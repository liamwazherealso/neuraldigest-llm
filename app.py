import logging
from datetime import datetime, timedelta
import os
from collections import defaultdict

import boto3 
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
import weaviate


logger = logging.getLogger()
DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
topics = ["business", "entertainment", "nation", "science", "technology", "world"]

config = {}

def news_by_topic(topic: str):
    """Get all articles from Weaviate by topic"""

    results = []

    LIMIT = 100
    offset = 0

    client = weaviate.Client(
        url=config["weaviate_url"], 
        auth_client_secret=weaviate.AuthApiKey(api_key=config["WEAVIATE_API_KEY"]), 
        additional_headers = {
            "X-OpenAI-Api-Key": config["OPENAI_API_KEY"]
        }
    )

    where_filter = {
      "path": ["published_date"],
      "operator": "Equal",
      "valueString": DATE
    }

    where_filter2 = {
      "path": ["topic"],
      "operator": "Equal",
      "valueString": topic
    }

    while True:
        result = (
          client.query
          .get("Article", ["title"])
          .with_limit(LIMIT)
          .with_offset(offset)
          .with_additional(["vector"])
          .with_where(where_filter)
          .with_where(where_filter2)
          .do()
        )

        result = result['data']['Get']['Article']

        results.extend(result)

        if len(result) < LIMIT:
            break

        offset += LIMIT
    return results

def summarize_article(topic: str, article_text: str):
    """Summarize article using OpenAI API & Langchain"""

    article_doc = Document(page_content=article_text, metadata={"source": str(0)})

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 20,
    )


    texts = text_splitter.split_documents([article_doc])

    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    return chain.run(texts)

def get_summarized_articles(articles: list[dict]) -> defaultdict:
    """Get summarized articles from Weaviate by topic"""
    summarized_articles = defaultdict(dict)
    

    for topic in topics:
        articles = news_by_topic(topic)

        i = 0
        print(topic)
        
        for article in articles:
            # TODO: Take care of in ETL
            if "text" not in article or not article["text"]:
                continue
                
            i += 1
            print("{}: {}".format(i, article["title"]))
            if article["title"] in summarized_articles[topic]:
                print("Skipping {}, already in dict".format(article["title"]))
                continue 
                
            article["text"] = summarize_article(DATE, topic, article["text"])
            summarized_articles[topic][article["title"]] = article

            if i >= 3:
                break

    return summarized_articles

def write_summ_article(article: dict, html: str) -> str:
    """Write summarized article to HTML"""
    article_html = f"""<h3>{article["title"].replace('.json', '')}</h3>
<p>{article["text"]}</p>
<a href=\"{article["url"]}\"> source </a>
"""
    
    html += article_html
    return html 
    
def write_section(articles: list[dict], topic: str):
    """Write topic section to HTML"""

    html=f"""<h1>{topic.capitalize()}</h1>\n<hr>\n"""
    
    for article in articles.values():
        html = write_summ_article(article, html)
    html += "\n"
    return html

def write_html():
    """Write HTML file"""

    summarized_articles = get_summarized_articles()

    article_html = f"""<!DOCTYPE html>
    <html lang="en-US">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>NeuralDigest Summary: {DATE}</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <link href="../style.css" rel="stylesheet">
    </head>
    <h1> Summary for News Articles: {DATE} </h1>
    <hr>
    <body>
    """

    for topic in topics:
        article_html += write_section(summarized_articles[topic], topic)

    article_html += "</html>"
    return article_html


def lambda_handler(event, _):
    if "LOG_LEVEL" in event and event["LOG_LEVEL"].lower() in [
        "debug",
        "info",
        "warning",
        "error",
        "critical",
    ]:
        logger.setLevel(level=getattr(logging, event["LOG_LEVEL"].upper()))
    else:
        logger.setLevel(level=logging.INFO)

    s3 = boto3.client('s3')
    file_name = f'articles/{DATE}.html'
    
    bucket_name = config["S3_BUCKET_NAME"]

    configVars = [
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "OPENAI_API_KEY",
    ]
    for conf in configVars:
        config[conf] = event[conf]

    try:
        # Write article_html to S3 bucket
        s3.put_object(Body=write_html(), Bucket=bucket_name, Key=file_name)
        print(f'Successfully written {file_name} to {bucket_name}')
    except Exception as e:
        print(f'Error writing {file_name} to {bucket_name}: {str(e)}')

