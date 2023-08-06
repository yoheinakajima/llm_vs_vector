# llm_vs_vector
Made a little script to speed and cost of classification via LLM or via vector embeddings

Currently tests the classification of 50 sentences into positive or negative using three approaches:
- ChatCompletion [gpt-3.5-turbo]
- Comparing vector embeddings (to positive and negative) [text-embedding-ada-002]
- Comparing vector embeddings (to positive and negative) [spacy]

Stats Tracked:
- Speed is tracked for all three methods.
- Distance to positive and negative is tracked for vector embedding methods
- Token count and cost is tracked for ChatCompletion and ada-002 vector embedding.

Here's the <a href="http://yoheinaka](https://twitter.com/yoheinakajima/status/1688032436788322304" target="_blank">Original tweet thread</a> about this.

Example output:
<img src="https://pbs.twimg.com/media/F21AulEb0AAPC5A?format=jpg&name=4096x4096" alt="Example output">



## Setup
`pip install -r requirements.txt`
`python -m spacy download en_core_web_md`
`export OPENAI_API_KEY=<your key here>`

## Usage
`python main.py`