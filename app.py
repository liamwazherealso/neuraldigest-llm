import logging
import json
from collections import defaultdict
from datetime import datetime, timedelta

import boto3
import weaviate

from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI



logger = logging.getLogger()
DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
ARTICLE_LIMIT = 3

topics = ["business", "entertainment", "nation", "science", "technology", "world"]

config = {}

def news_by_topic(topic: str):
    """Get all articles from Weaviate by topic"""

    results = []

    offset = 0

    client = weaviate.Client(
        url=config["WEAVIATE_URL"], 
        auth_client_secret=weaviate.AuthApiKey(api_key=config["WEAVIATE_API_KEY"])
    )

    REQ_LIMIT = 100

    if ARTICLE_LIMIT < REQ_LIMIT:
        REQ_LIMIT = ARTICLE_LIMIT
        
    where_filter = {
        "operator": "And",
        "operands": [{
            "path": ["published_date"],
            "operator": "Equal",
            "valueString": DATE
            },{
            "path": ["topic"],
            "operator": "Equal",
            "valueString": topic
        }]
    }

    while True:
        result = (
          client.query
          .get("Article", ["title", "text", "url"])
          .with_limit(REQ_LIMIT)
          .with_offset(offset)
          .with_additional(["vector"])
          .with_where(where_filter)
          .do()
        )

        result = result['data']['Get']['Article']

        results.extend(result)

        if len(result) < REQ_LIMIT or len(results) >= ARTICLE_LIMIT:
            break

        offset += REQ_LIMIT
    return results

def summarize_article(article_text: str):
    """Summarize article using OpenAI API & Langchain"""

    article_doc = Document(page_content=article_text, metadata={"source": str(0)})

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 20,
    )


    texts = text_splitter.split_documents([article_doc])

    llm = OpenAI(temperature=0, openai_api_key=config["OPENAI_API_KEY"])
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    return chain.run(texts)

def get_summarized_articles() -> defaultdict:
    """Get summarized articles from Weaviate by topic"""
    summarized_articles = defaultdict(dict)
    

    for topic in topics:
        articles = news_by_topic(topic)

        logging.debug(topic)
        
        for article in articles:
            # TODO: Take care of in ETL
            if "text" not in article or not article["text"]:
                logging.debug("Skipping {}, no text".format(article["title"]))
                continue
                
            if article["title"] in summarized_articles[topic]:
                logging.debug("Skipping {}, already in dict".format(article["title"]))
                continue 
                
            article["text"] = summarize_article(article["text"])
            summarized_articles[topic][article["title"]] = article


    return summarized_articles

def write_summ_article(article: dict, html: str) -> str:
    """Write summarized article to HTML"""
    logging.debug("Starting: write_summ_article for {}".format(article["title"]))
    article_html = f"""<h3>{article["title"].replace('.json', '')}</h3>
<p>{article["text"]}</p>
<a href=\"{article["url"]}\"> source </a>
"""
    
    html += article_html
    logging.debug("Finished: write_summ_article for {}".format(article["title"]))
    return html 
    
def write_section(articles: list[dict], topic: str):
    """Write topic section to HTML"""

    logging.debug(f"Starting: write_section for {topic} with {len(articles)} articles")

    html=f"""<h1>{topic.capitalize()}</h1>\n<hr>\n"""
    
    for article in articles.values():
        html = write_summ_article(article, html)
    html += "\n"

    logging.debug(f"Finished: write_section for {topic}")
    return html

def write_html():
    logging.debug("Staring: write_html")
    """Write HTML file"""

    summarized_articles = get_summarized_articles()

    article_html = f"""<!DOCTYPE html>
    <html lang="en-US">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
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
    
    logging.debug("Finished: write_html")
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

    logging.debug("Initiating lambda_handler")
    s3 = boto3.client('s3')
    file_name = f'articles/{DATE}.html'
    
    bucket_name = event["S3_BUCKET_NAME"]

    
    configVars = [
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "OPENAI_API_KEY",
    ]
    for conf in configVars:
        try:
            config[conf] = event[conf]
        except KeyError:
            logging.warning(f"Missing event variable: {conf}")
    try:
        # Write article_html to S3 bucket
        s3.put_object(Body=write_html(), Bucket=bucket_name, Key=file_name, ContentType='text/html')
        logging.info(f'Successfully written {file_name} to {bucket_name}')
    except Exception as e:
        logging.warning(f'Error writing {file_name} to {bucket_name}: {str(e)}')
    
    logging.debug("Retrieving and modifying articles.json")
    articles_json_key = "articles.json"
    response = s3.get_object(Bucket=bucket_name, Key=articles_json_key)
    content = response['Body'].read().decode('utf-8')
    articles = json.loads(content)

    if DATE not in articles:
        articles.insert(0, DATE)

    modified_content = json.dumps(articles)

    s3.put_object(Bucket=bucket_name, Key=articles_json_key, Body=modified_content)

