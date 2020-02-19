# AI_Trading
This project builds a deep learning model to classify the sentiment of messages from StockTwits, a social network for investors and traders. This model is able to identify if the information conveyed by a message is positive or negative. The sentimental signal can be further applied for stock prediction along with other information reflected by the market.

## Getting Started
Train the model by 
```
{git_repository}/nlp_trading/src/run_proj estimation
```
The arguments need to be specified include 'data_directory' and 'output_directory'. Once the model is trained and stored in output directory, it can be used for prediction with command 
```
{git_repository}/nlp_trading/src/run_proj prediction --text TEXT
```
## Prerequisites
All the required packages and their versions can be found in requirement.txt.

## Acknowledgments
This production code is inspired by the projects in Udacity online program 'AI for Trading'.
