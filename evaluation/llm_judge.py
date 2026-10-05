import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


class LLMJudge:

    def __init__(
        self,
        model="gpt-4o-mini"
    ):

        self.llm = ChatOpenAI(
            model=model,
            temperature=0
        )


    # ==========================================================
    # Evaluate one RAG response
    # ==========================================================

    def evaluate(
        self,
        question,
        context,
        answer
    ):

        prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer using ONLY the supplied context.

QUESTION:
{question}

CONTEXT:
{context}

GENERATED ANSWER:
{answer}


Evaluate the answer on three dimensions.

1. ANSWER RELEVANCE

Does the generated answer directly answer the question?

Score:
1 = completely irrelevant
2 = mostly irrelevant
3 = partially relevant
4 = mostly relevant
5 = completely relevant


2. FAITHFULNESS

Are the claims in the generated answer supported by the supplied context?

Score:
1 = completely unsupported
2 = mostly unsupported
3 = partially supported
4 = mostly supported
5 = completely supported


3. CONTEXT RELEVANCE

Does the supplied context contain information that is useful for answering
the question?

Score:
1 = completely irrelevant
2 = mostly irrelevant
3 = partially relevant
4 = mostly relevant
5 = highly relevant


Return ONLY valid JSON.

The JSON must have exactly this structure:

{{
    "answer_relevance": <integer from 1 to 5>,
    "faithfulness": <integer from 1 to 5>,
    "context_relevance": <integer from 1 to 5>,
    "reason": "<short explanation>"
}}
"""


        response = self.llm.invoke(
            prompt
        )


        content = response.content.strip()


        # ======================================================
        # Remove markdown JSON fences if returned
        # ======================================================

        if content.startswith(
            "```json"
        ):

            content = content[
                7:
            ]

            if content.endswith(
                "```"
            ):

                content = content[
                    :-3
                ]


        elif content.startswith(
            "```"
        ):

            content = content[
                3:
            ]

            if content.endswith(
                "```"
            ):

                content = content[
                    :-3
                ]


        content = content.strip()


        # ======================================================
        # Parse JSON
        # ======================================================

        try:

            evaluation = json.loads(
                content
            )

        except json.JSONDecodeError:

            return {
                "answer_relevance": 0,
                "faithfulness": 0,
                "context_relevance": 0,
                "reason":
                    "Judge returned invalid JSON."
            }


        return evaluation