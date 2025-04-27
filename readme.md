# QEX - Query Evaluator eXtended
This project implements RAG systems in context of the answer evaluation using RAGAS metrics.

## Core features
- Creation of RAG system via dedicated factory.
- Sophisticated builder for a query to support multiple customizable promp engineering techniques.
- Dynamically updated vectorstore with dedicated provider for both sparse and dense retrievers.
 - Evaluator class, capable of building queries with their ground truths into evaluation datasets and later evaluating them using RAGAS metrics.
 - Operation logging to the console and tracking the logs inside a singleton logger.

 ## Design choices
 In order to achieve strong SOLID and also sophisticated and effective programming I have decided on following design choices:
 - Lazy loading for both retrievers in vectorstore, RAGs' chains and also for all variables related to evaluation. Thanks to that if there is no need for any of those they won't be created/updated which saves memory and improves efficiency. To do that I used a property in a class which houses the extended lazy loading logic and also private attributes inside the classes, which house the actual value.
 - Factory design pattern for RAGs. I create a specific type of RAG via specially prepared methods in my factory.
 - Proxy design pattern for retirever in the RAG. The retrievers are accessed via lazy loading with rebuilding in case anything in documents folder changes, but upon calling property it would return fixed reference. This approach made it so RAG would always have same retriever which is not correct with state of documents. To circumvent that I have used a RetrieverProxy which takes a callable which returns the property value. Proxy when called with standard retriever methods will first get a retriever from callable. This way the dynamic characteristic of retriever logic is preserved and works as intended.
 - Builder design pattern for creating the query. In order to achieve complete flexibility with creating specific query variations according to prompt engineering techniques I have decided to use a builder pattern, which allows for robust customization of query including providing ground truth and also selecting any technique and for some adjusting its parameters.
 