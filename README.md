[![REUSE status](https://api.reuse.software/badge/github.com/SAP/langchain-integration-for-sap-hana-cloud)](https://api.reuse.software/info/github.com/SAP/langchain-integration-for-sap-hana-cloud)

# LangChain integration for SAP HANA Cloud

## About this project

Integrates LangChain with SAP HANA Cloud to make use of vector search, knowledge graph, and further in-database capabilities as part of LLM-driven applications.

## Requirements and Setup

### Prerequisites

- **Python Environment**: Ensure you have Python 3.9 or higher installed.
- **SAP HANA Cloud**: Access to a running SAP HANA Cloud instance.


### Installation

Install the LangChain SAP HANA Cloud integration package using `pip`:

```bash
pip install -U langchain-hana
```

### Setting Up Vectorstore

The `HanaDB` class is used to connect to SAP HANA Cloud Vector Engine.

Here’s how to set up the connection and initialize the vector store:

```python
from langchain_hana import HanaDB
from hdbcli import dbapi

# use a LangChain Embeddings class
embeddings = ...  

# Establish the SAP HANA Cloud connection
connection = dbapi.connect(
    address="<hostname>",
    port=3<NN>MM,
    user="<username>",
    password="<password>"
)

# Initialize the HanaDB vector store
vectorstore = HanaDB(
    connection=connection,
    embeddings=embeddings,
    table_name="<table_name>"  # Optional: Default is "EMBEDDINGS"
)

```


## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/langchain-integration-for-sap-hana-cloud/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/langchain-integration-for-sap-hana-cloud/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2025 SAP SE or an SAP affiliate company and langchain-integration-for-sap-hana-cloud contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/langchain-integration-for-sap-hana-cloud).
