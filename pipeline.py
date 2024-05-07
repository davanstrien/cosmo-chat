import os
import random

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadHubDataset,
    StepInput,
    StepOutput,
    step,
)
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import KeepColumns
from dotenv import load_dotenv
from rich import print

# Model Configuration

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

QS_GEN_BATCH_SIZE = 20
ANSWER_GEN_BATCH_SIZE = 10


@step(inputs=["audience"], outputs=["audience"])
def clean_audience(inputs: StepInput) -> StepOutput:
    for input in inputs:
        if input["audience"] == "grade_school_students":
            input["audience"] = "grade school student"
        if input["audience"] in ["college_studnets", "college_students"]:
            input["audience"] = "college student"
    yield inputs


def cosmopedia_to_question_prompt(text, audience) -> str:
    return f""""You will play a {audience}.
Based on some text I will show you, please write a question about the topic discussed which could be asked by a {audience}.
The question should focus on topics discussed in the text. You should write the question assuming the student has some familiarity with the topic but not the text itself.
<text>\n
{text.strip()}
</text>
If the text includes analogies, examples, metaphors etc., you should not include them in the question unless it is reasonable to assume these would be commonly used in educational materials for a {audience}.
For example, if the text discusses the concept of gravity using the analogy of a bowling ball on a trampoline, you should not include the analogy in the question unless it is reasonable to assume that a {audience} would be familiar with this analogy.
Please generate questions that:
- Require critical thinking and cannot be easily answered by quoting the text verbatim.
- Probe for deeper understanding and conceptual knowledge of the topic.
- Are concise and suitable in complexity for a {audience}.
Return only the question you would ask, not the text itself or any other information.
"""


@step(inputs=["text", "audience"], outputs=["prompt_for_first_question"])
def prepare_prompt_for_first_question(inputs: StepInput) -> StepOutput:
    for input in inputs:
        first_question_prompt = cosmopedia_to_question_prompt(
            input["text"], input["audience"]
        )
        input["prompt_for_first_question"] = first_question_prompt
    yield inputs


def format_to_generate_second_question(questions_and_answer_messages, audience) -> str:
    student_understanding = [
        "a very poor understanding of the topic. This student may be confused or have misconceptions about the topic which will be expressed in their question. Example: a question that reveals a fundamental misunderstanding of a key concept.",
        "a poor grasp of the topic and wants to clarify their understanding. Example: a question that seeks clarification on a specific point mentioned in the previous answer.",
        "a good understanding of the topic. They are likely to follow up about a specific component of the answer given. Example: a question that explores a subtopic or asks for elaboration on a particular aspect of the answer.",
        f"a very deep understanding of the topic and wants to explore the topic further. This student may ask questions beyond what would be expected from their level of study as a {audience}. They may make connections to other topics or ask about advanced concepts. Example: a question that draws parallels to related subjects or inquires about the implications of the discussed ideas.",
    ]
    student_understanding = random.choice(student_understanding)
    messages_formatted = ""
    for message in questions_and_answer_messages:
        if message["role"] == "user":
            messages_formatted += f"Q: {message['content']}\n"
        if message["role"] == "system":
            messages_formatted += f"A: {message['content']}\n"
    return f"""Based on the conversation below, write a follow-up question from a student.
    <conversation>
    {messages_formatted}
    </conversation>
    Remember the student is a {audience} with {student_understanding}. 
    Generate a follow-up question that builds upon the previous answer.
    If the response from the model is beyond what would be expected from a {audience}, reflect this in the question.
    Please ensure the question is relevant to the main topic and avoid tangential or unrelated questions.
    Just respond with the question.
    """


@step(
    inputs=["first_question", "first_answer", "audience"],
    outputs=["prompt_for_second_question"],
)
def prepare_follow_up_question(inputs: StepInput) -> StepOutput:
    for input in inputs:
        messages = [
            {"role": "user", "content": input["first_question"]},
            {"role": "system", "content": input["first_answer"]},
        ]
        follow_up_question_prompt = format_to_generate_second_question(
            messages, input["audience"]
        )
        input["prompt_for_second_question"] = follow_up_question_prompt
    yield inputs


@step(
    inputs=[
        "first_question",
        "first_answer",
        "follow_up_question",
    ],
    outputs=["prompt_for_follow_up_answer"],
)
def prompt_for_follow_up_answer(inputs: StepInput) -> StepOutput:
    for input in inputs:
        messages = [
            {"role": "user", "content": input["first_question"]},
            {"role": "system", "content": input["first_answer"]},
            {"role": "user", "content": input["follow_up_question"]},
        ]
        input["prompt_for_follow_up_answer"] = messages
    yield inputs


def format_final_response(
    questions_and_answer_messages: list[dict[str, str]], audience
) -> str:
    messages_formatted = ""
    for message in questions_and_answer_messages:
        if message["role"] == "user":
            messages_formatted += f"Q: {message['content']}\n"
        if message["role"] == "system":
            messages_formatted += f"A: {message['content']}\n"
    return f"""Based on the conversation below, generate a thoughtful and engaging final question that a user could ask to further the discussion.
    <conversation>
    {messages_formatted}
    </conversation>
    Guidelines for generating the question:

    The question should be a natural continuation of the conversation, either following up on a previous point or exploring a related aspect of the topic.
    The question should be designed to challenge the model's ability to provide a comprehensive, insightful, or explanatory response.
    The question should encourage the model to demonstrate its knowledge, reasoning skills, or ability to clarify complex concepts.
    The question should be open-ended and thought-provoking, avoiding simple yes/no or factual answers.
    The question should be concise, clear, and well-formulated.
    The question may require the model to consider multiple perspectives, analyze trade-offs, or explore hypothetical scenarios related to the topic.
    The question could ask the model to draw connections between the current topic and other relevant concepts, fields, or real-world applications.
    The question might challenge the model to provide examples, analogies, or case studies to support its explanations or arguments.
    The question could encourage the model to discuss potential future developments, implications, or challenges related to the topic.
    The question should be realistic and resemble something a curious and engaged {audience} might ask in the given context.
    Avoid generating redundant or repetitive questions that have already been addressed in the conversation.
    Please return only the generated question, without any additional text or formatting.
    """


@step(
    inputs=[
        "first_question",
        "first_answer",
        "follow_up_question",
        "follow_up_answer",
        "audience",
    ],
    outputs=["final_question_prompt"],
)
def create_final_question(inputs: StepInput) -> StepOutput:
    for input in inputs:
        messages = [
            {"role": "user", "content": input["first_question"]},
            {"role": "system", "content": input["first_answer"]},
            {"role": "user", "content": input["follow_up_question"]},
            {"role": "system", "content": input["follow_up_answer"]},
        ]
        final_question_prompt = format_final_response(messages, input["audience"])
        input["final_question_prompt"] = final_question_prompt
    yield inputs


@step(
    inputs=[
        "first_question",
        "first_answer",
        "follow_up_question",
        "follow_up_answer",
        "final_question",
    ],
    outputs=["prompt_for_final_answer"],
)
def prompt_for_final_answer(inputs: StepInput) -> StepOutput:
    for input in inputs:
        messages = [
            {"role": "user", "content": input["first_question"]},
            {"role": "system", "content": input["first_answer"]},
            {"role": "user", "content": input["follow_up_question"]},
            {"role": "system", "content": input["follow_up_answer"]},
            {"role": "user", "content": input["final_question"]},
        ]
        input["prompt_for_final_answer"] = messages
    yield inputs


@step(
    inputs=[
        "first_question",
        "first_answer",
        "follow_up_question",
        "follow_up_answer",
        "final_question",
        "final_answer",
    ],
    outputs=["dialogue"],
)
def format_full_dialogue(inputs: StepInput) -> StepOutput:
    for input in inputs:
        messages = [
            {"role": "user", "content": input["first_question"]},
            {"role": "system", "content": input["first_answer"]},
            {"role": "user", "content": input["follow_up_question"]},
            {"role": "system", "content": input["follow_up_answer"]},
            {"role": "user", "content": input["final_question"]},
            {"role": "system", "content": input["final_answer"]},
        ]
        input["dialogue"] = messages
    yield inputs


with Pipeline(name="cosmo-chat") as pipeline:
    question_generator_llm = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
        model_display_name="meta-llama/Meta-Llama-3-70B-Instruct",
        api_key=HF_TOKEN,
    )
    answer_generator_llm_llama = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
        model_display_name="meta-llama/Meta-Llama-3-70B-Instruct",
        api_key=HF_TOKEN,
    )

    load_hub_dataset = LoadHubDataset(
        name="load_dataset",
    )
    format_audience = clean_audience(
        name="format_audience",
    )
    create_initial_prompt = prepare_prompt_for_first_question(
        name="create_initial_prompt"
    )
    load_hub_dataset.connect(format_audience)
    format_audience.connect(create_initial_prompt)
    first_question_generation = TextGeneration(
        name="first_question_generation",
        llm=question_generator_llm,
        input_mappings={"instruction": "prompt_for_first_question"},
        output_mappings={
            "generation": "first_question",
        },
        input_batch_size=QS_GEN_BATCH_SIZE,
    )
    create_initial_prompt.connect(first_question_generation)
    first_answer = TextGeneration(
        name="first_answer_generator",
        llm=answer_generator_llm_llama,
        input_mappings={"instruction": "first_question"},
        output_mappings={"generation": "first_answer"},
        input_batch_size=ANSWER_GEN_BATCH_SIZE,
    )
    first_question_generation.connect(first_answer)
    follow_up_question = prepare_follow_up_question(name="follow_up_question")
    first_answer.connect(follow_up_question)
    second_question_generation = TextGeneration(
        name="second_question_generation",
        llm=question_generator_llm,
        input_mappings={"instruction": "prompt_for_second_question"},
        output_mappings={"generation": "follow_up_question"},
        input_batch_size=QS_GEN_BATCH_SIZE,
    )
    follow_up_question.connect(second_question_generation)
    follow_up_prompt = prompt_for_follow_up_answer(name="follow_up_prompt")
    second_question_generation.connect(follow_up_prompt)
    follow_up_answer = TextGeneration(
        name="follow_up_answer_generator",
        llm=answer_generator_llm_llama,
        input_mappings={"instruction": "prompt_for_follow_up_answer"},
        output_mappings={"generation": "follow_up_answer"},
        input_batch_size=ANSWER_GEN_BATCH_SIZE,
    )
    follow_up_prompt.connect(follow_up_answer)
    final_question = create_final_question(name="final_question")
    follow_up_answer.connect(final_question)
    final_question_generation = TextGeneration(
        name="final_question_generation",
        llm=question_generator_llm,
        input_mappings={"instruction": "final_question_prompt"},
        output_mappings={"generation": "final_question"},
        input_batch_size=QS_GEN_BATCH_SIZE,
    )
    final_question.connect(final_question_generation)
    final_prompt = prompt_for_final_answer(name="final_prompt")
    final_question_generation.connect(final_prompt)
    final_answer_llama = TextGeneration(
        name="final_answer_generator",
        llm=answer_generator_llm_llama,
        input_mappings={"instruction": "prompt_for_final_answer"},
        output_mappings={"generation": "final_answer"},
        input_batch_size=ANSWER_GEN_BATCH_SIZE,
    )
    final_prompt.connect(final_answer_llama)
    full_dialogue = format_full_dialogue(name="full_dialogue")
    final_answer_llama.connect(full_dialogue)
    keep_columns = KeepColumns(
        name="keep_columns",
        columns=["dialogue", "text", "audience", "final_question_prompt", "seed_data"],
        output_mappings={"text": "cosmopedia_text"},
    )
    full_dialogue.connect(keep_columns)

if __name__ == "__main__":
    disti = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": "davanstrien/cosmopedia_sample_10",
                "split": "train",
            },
            "first_question_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 500,
                        "do_sample": True,
                        "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                    }
                }
            },
            "first_answer_generator": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 2000,
                        "do_sample": True,
                        "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                    }
                }
            },
            "second_question_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 500,
                        "do_sample": True,
                        "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                    }
                }
            },
            "follow_up_answer_generator": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 2000,
                        "do_sample": True,
                        "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                    }
                }
            },
            "final_question_generation": {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 2000},
                    "do_sample": True,
                    "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                }
            },
        },
        use_cache=True,
    )
    disti.push_to_hub("davanstrien/cosmochat", private=True)
