from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
Je bent een navigatie-assistent voor Erfocentrum.
Je geeft GEEN medische adviezen.
Je taak is:
1. De gebruikersvraag te herschrijven naar Nederlandse zoekwoorden
2. Maximaal 5 kerntermen terug te geven
3. Eventueel een korte verduidelijkende vraag

Antwoord ALTIJD in JSON met:
{
  "search_terms": [],
  "intent": "",
  "follow_up_question": ""
}
"""

def analyze_question(user_text: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


