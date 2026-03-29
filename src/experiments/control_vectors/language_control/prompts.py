COHERENCE_PROMPT_TEMPLATE = """
[Role and Goal]
You are an impartial judge. Your goal is to evaluate the coherence of a text continuation based on the preceding context.

[Core Task]
Select the most appropriate category that describes how well the continuation fits with the preceding sentences. The evaluation should be based on semantic relatedness, logical flow, and overall coherence.

For example, if the context is about a story involving a dog and the continuation mentions the dog doing something related to the story, it would be considered coherent. If the continuation seem to about some totally different topic without any connection to the sentences, it would be incoherent.

[Evaluation Categories]
  1. The continuation is completely unrelated to the previous sentences with no semantically resembling words, concepts, or structure indicating of relatedness.
  2. The continuation is related to the overarching topic or themes of the previous sentences (e.g., 'dogs', 'shopping') but fails to connect to the specific events or characters of the sentences.
  3. The continuation correctly identifies and uses specific elements (characters, objects, locations) from the previous sentences. However, it contains some major logical inconsistencies, e.g., by mixing up the subject of some actions or presenting conflicting facts about the elements.
  4. The continuation provides a coherent continuation to the previous sentences without illogical or incosistent elements.

[Special Instructions]
The continuation may be in a different language than the context. Evaluate the coherence of the meaning, not the language itself.

[Input]
Previous sentences:
{context}

Continuation:
{continuation}

[Output Instructions]
Provide your answer in a single JSON object with keys, "reasoning" and "answer", where "reasoning" is a short explanation of your evaluation and "answer" is the number of the category which matches best with your evaluation for the continuation.

Format:
{{"reasoning": "Your short reasoning here", "answer": number}}

"""

FLUENCY_PROMPT_TEMPLATE = """
[Role and Goal]
You are a native-level Finnish proofreader and an impartial judge. Your goal is to evaluate the Finnish usage of a text snippet based on its Finnish grammatical correctness and Finnish fluency.

[Core Task]
Select the most appropriate category that describes whether the text is grammatically correct and fluent in Finnish.

For example, if the text is grammatically correct and idiomatic, it would be considered fluent. If it is in different language or contains significant grammatical errors or unnatural phrasing, it would be considered less fluent.

[Evaluation Categories]
  1. The text is grammatically malformed to the point of being incomprehensible (gibberish), is not in Finnish, or contains only a few Finnish words in a sea of another language.
  2. The text is understandable and mostly in Finnish, but contains many significant grammatical errors or words from other languages. Grammatical errors may include fundamentally wrong verb conjugations or noun inflections, severe phrasing issues, or highly repetitive words or phrases.
  3. The text is fully in Finnish and grammatically correct for the most part, but contains minor, non-critical errors. Examples include unnatural phrasing that suggests a literal translation from another language (calque), occasional incorrect word inflections, or awkward word choices.
  4. The text is fully in Finnish and grammatically perfect, idiomatic, and reads as if written by a native Finnish speaker. All word choices, inflections, and structures are natural.

[Special Instructions]
Note the following things in your evaluation:
- The text is truncated so do not consider the fact that the text is not complete / might end abruptly in your evaluation.
- The text may be completely in different language than Finnish, making category 1 the only applicable one.

[Input]
{continuation}[truncated]

[Output Instructions]
Provide your answer in a single JSON object with keys, "reasoning" and "answer", where "reasoning" is a short explanation of your evaluation and "answer" is the number of the category which matches best with your evaluation for the text.

Format:
{{"reasoning": "Your short reasoning here", "answer": number}}

"""
