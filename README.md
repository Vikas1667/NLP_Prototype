# NLP_Prototype





### Architechture

### NLP Techniques
1. Sense2vec
2. Subword Techniques
3. Semantic search 


#### onpremise testing 
1. Airflow and Spark 
2. MongoDB
3. Solr 
4. Pysql 
5. Docker 


#### References
1. https://medium.com/analytics-vidhya/subword-techniques-for-neural-machine-translation-f55e4506a728

## Usecases
1. Scraping the news articles 
    1) EL paiso -spainish english
    2) Daily Mail
    3) Investing.com
 
2. Sentiment analysis
    1. sentiment analysis : Finbert  
    2. Emotion and Feeling extraction

3. Income statement and financial 
    1. PDF with spark and airflow

4. Extraction of stock data 
5. Audio File usecases
6. Stock tickets extraction

References
1. https://towardsdatascience.com/how-to-build-a-data-extraction-pipeline-with-apache-airflow-fa83cb8dbcdf
2. https://github.com/dmesquita/airflow-tutorial
3. https://towardsdatascience.com/web-scraping-for-accounting-analysis-using-python-part-1-b5fc016a1c9a
4. https://medium.com/@jan_5421/extracting-financial-statements-from-sec-filings-xbrl-to-json-f83542ade90
5. https://jovian.ai/omprakashp014909/extracting-stock-data-using-python-libraries
6. https://towardsdatascience.com/how-to-extract-keywords-from-audio-files-with-natural-language-processing-nlp-3084ceb951c9
7. https://randerson112358.medium.com/web-scraping-stock-tickers-using-python-3e5801a52c6d



## Cloud 

1. Services
2. Usage and Recipes
3. Usecases
4. Costing 
5. Performance Comparision


### 1. Services
##### AWS 
1. Sagemaker
2. Amazon Personalize
    1. Data preparation
3. Amazon Wrangler 

#####
1. https://aws.amazon.com/blogs/machine-learning/building-a-customized-recommender-system-in-amazon-sagemaker/
2. https://aws.amazon.com/getting-started/hands-on/semantic-content-recommendation-system-amazon-sagemaker/3
3. [Accelerate and improve recommender system training and predictions using Amazon SageMaker Feature Store](https://aws.amazon.com/blogs/machine-learning/accelerate-and-improve-recommender-system-training-and-predictions-using-amazon-sagemaker-feature-store/)
4. https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html
5. https://aws.amazon.com/sagemaker/data-wrangler/
6. https://docs.aws.amazon.com/personalize/latest/dg/data-prep.html


Use cases
1. Recommendation Systems 

[Creating a recommendation engine using Amazon Personalize](https://aws.amazon.com/blogs/machine-learning/creating-a-recommendation-engine-using-amazon-personalize/)
[Amazon Personalize can now create up to 50% better recommendations for fast changing catalogs of new products and fresh content](https://aws.amazon.com/blogs/machine-learning/amazon-personalize-can-now-create-up-to-50-better-recommendations-for-fast-changing-catalogs-of-new-products-and-fresh-content/)
[How Restaurants Can Win Over Generation Z With Text To Order](https://aws.amazon.com/blogs/industries/how-restaurants-can-win-over-generation-z-with-text-to-order/)
https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada







## GCP
Services 
    1. Vertex AI

Pricing

https://cloud.google.com/vertex-ai/pricing#asia-pacific
https://cloud.google.com/products/ai
https://developers.google.com/machine-learning/recommendation/overview


##
AWS



Azure- ML
https://azure.microsoft.com/en-in/services/machine-learning/#product-overview

https://docs.microsoft.com/en-in/azure/machine-learning/

Pricing
https://cloud.google.com/vertex-ai/pricing#asia-pacific


OCR services
https://ricciuti-federico.medium.com/how-to-compare-ocr-tools-tesseract-ocr-vs-amazon-textract-vs-azure-ocr-vs-google-ocr-ba3043b507c1
AWS
Sagemaker
https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html
 
GCP - ML
https://cloud.google.com/products/ai

services 
Vertex AI
 
 
 
Azure- ML

## Azure OCR pipeline for PDF to Knowledge Graph 
1. [Extracting Key-Value Pairs from PDF Documents Using Deep Learning](https://medium.com/@faysal887/extracting-key-value-pairs-from-pdf-documents-using-deep-learning-fd79f4582a86)

https://azure.microsoft.com/en-in/services/machine-learning/#product-overview 
https://docs.microsoft.com/en-in/azure/machine-learning/
 
Compartive studies 

1. OCR services across various cloud platforms

1. https://ricciuti-federico.medium.com/how-to-compare-ocr-tools-tesseract-ocr-vs-amazon-textract-vs-azure-ocr-vs-google-ocr-ba3043b507c1

Image Classification, Object Detection 

Model : CNN Architechture 
Use cases Overview: AWS site explore(CNN)  
Key Observations, Issues, Roadmap with links


Text Classification, Text Embedding

Model : BERT Architechture 
Use cases Overview: AWS site explore(BERT)  
Key Observations and Roadmap 

eg: Reading Various text format 

Algorthms 
1) Tokenizer : Wordpiece tokenizer 
eg : tokenizer.tokenize(Komalization):-> Komal + ##zation

2) Attention Vectors 
self attention , Multi head attention 
  i) Sentiment Analysis : Additive attention 

  
issues 
Reading from local 
https://stackoverflow.com/questions/62472238/autotokenizer-from-pretrained-fails-to-load-locally-saved-pretrained-tokenizer

Tabular Regression,Tabular Classification: Sumit
https://aws.amazon.com/blogs/machine-learning/new-built-in-amazon-sagemaker-algorithms-for-tabular-data-modeling-lightgbm-catboost-autogluon-tabular-and-tabtransformer/


