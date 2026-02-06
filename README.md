# Erfocentrum Search & Navigation Prototype

A study prototype created to enhance users with different levels of health literacy's access to genetic health information is included in this repository. The prototype adds helpful features like typo tolerance, synonym support and optional AI-supported navigation to a basic keyword-based search interface.

The system is **informational only** and was created and assessed in an academic setting. It doesn't offer diagnosis, treatment suggestions or medical advice.

## Project Context and Motivation

Spelling mistakes, unfamiliar vocabulary and difficulty formulating queries are some of the reasons why users with lower health literacy frequently struggle while looking for medical information. This prototype looks into whether helpful search interface elements might enhance a genetic health information website's perceived usability, search efficiency and information findability.

The website of the Dutch national information center for genetics and hereditary disorders, Erfocentrum, served as the model. This system's content is entirely synthetic and meant just for demonstration and research.
## Key Features

- Baseline lexical keyword search
- Typo tolerance and spelling suggestions
- Synonym- and theme-based query expansion
- Optional AI-supported navigation for natural language input
- Transparent fallback to baseline search when AI is unavailable
- Glossary support for medical terminology
- Informational-only design without medical interpretation

## Technology Stack

- Python 3.10+
- Shiny for Python
- Pandas and NumPy
- OpenAI API (optional, for AI-supported navigation)


## Repository Structure

```
.
├── app.py                  # Main Shiny application
├── llm_navigator.py        # AI-supported navigation logic
├── data/
│   ├── dsp_dataset_erfocentrum_v1.csv
│   ├── synonyms.csv
│   ├── themes.csv
│   ├── glossary.csv
│   └── hl_support_blocks.csv
├── www/
│   ├── styles.css
│   └── assets/
├── requirements.txt
└── README.md
```


## Installation

1. Clone the repository:
```
git clone https://github.com/yasminaelka/erfocentrum_prototype.git
cd erfocentrum_prototype
```

2. (Recommended) Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

---

## Running the Application

To start the prototype locally, run:
```
shiny run app.py
```

The application starts a local web server.  
Open the URL shown in the terminal in your browser to interact with the prototype.


## AI-Supported Navigation (Optional)

The AI-supported navigation component uses the OpenAI API and is optional.

To enable AI navigation, set an environment variable:
```
export OPENAI_API_KEY="your_api_key_here"
```

If no API key is provided:
- the system automatically falls back to baseline search
- all core functionality remains available.

This design ensures robustness, transparency, and reduced dependence on external services.


## Data Usage and Ethics

- All datasets included in this repository are synthetic and created specifically for this research.
- No real medical data, patient data, or personal user data is processed or stored.
- The system does not log user queries beyond the local runtime session.
- The prototype is intended for research and demonstration purposes only.


## Evaluation and Replicability

The prototype was evaluated using an in-person, within-subject A/B usability study. Participants completed predefined search tasks using both a baseline interface and an enhanced interface with navigation support.

Evaluation focused on:
- task success,
- time to successful completion,
- number of search attempts,
- perceived usability.

All datasets, configuration files, and source code required to replicate the system and study setup are included in this repository. Detailed evaluation procedures and materials are described in the accompanying research report and appendices.


## Sustainability, Extensibility, and Reuse

Because the system is CSV-driven and modular, it can be extended to other health topics or informative websites.Decoupling AI functionality from the baseline system allows for deployment without relying on external APIs.
Future studies on search interfaces, AI-assisted navigation or health literacy can make use of or modify the prototype.
Since the goal of this project is usability evaluation rather than production deployment, automated unit tests are not included.

## Limitations

- This is a research prototype, not a production-ready system.
- No live or publicly hosted demo is provided due to API key requirements and ethical considerations.
- The AI component is limited to query reformulation and navigation support and does not generate medical explanations.
- Performance and scalability were not optimized beyond what was required for usability testing.


## License and Usage

This project is provided for academic and research purposes only.  
Reuse or adaptation should maintain the informational-only nature of the system and comply with relevant ethical and legal guidelines.


