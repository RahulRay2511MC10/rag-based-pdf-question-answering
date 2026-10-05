from langchain_openai import ChatOpenAI


class QueryRewriter:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )

    def rewrite(
        self,
        question,
        chat_history
    ):

        # If there is no previous conversation,
        # the question is already standalone.
        if not chat_history:

            return question

        history_text = "\n".join(
            chat_history
        )

        prompt = f"""
You are a query rewriting assistant.

Your task is to rewrite the user's latest question
into a standalone question that can be understood
without the previous conversation.

Use the conversation history to resolve:
- pronouns such as he, she, it, they
- references such as this, that, previous question
- omitted subjects
- follow-up references

Do not answer the question.

Return only the rewritten standalone question.

Conversation history:
{history_text}

Latest question:
{question}
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content.strip()