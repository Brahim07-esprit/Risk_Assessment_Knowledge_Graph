import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client


def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is to extract the key concepts (and non-personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, breaking them down into simpler concepts where necessary. "
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, framework, process, standard, regulation, misc]\n"
        "For risk management-specific content, make sure to include all types of risks, frameworks, standards, mitigation techniques, "
        "and other relevant terminologies.\n"
        "Assign the importance of each concept based on the contextual significance, including the frequency of mention, the level of detail provided, "
        "and its overall contribution to understanding risk management concepts.\n"
        "Format your output as a list of JSON with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": "The concept",\n'
        '       "importance": "The contextual importance of the concept on a scale of 1 to 5 (5 being the highest)",\n'
        '       "category": "The type of concept (e.g., event, framework, standard, etc.)",\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
        "Consider also that technical terms such as 'COSO Framework', 'ISO 31000', 'Risk Appetite', or 'Basel III' "
        "should be properly categorized into appropriate categories like framework, standard, or concept."
    )

    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))
    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```). Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n\n"
        
        "Thought 1: While traversing through each sentence, think about the key terms mentioned in it.\n"
        "\t- Terms may include object, entity, location, organization, person, condition, acronym, documents, service, concept, "
        "framework, process, standard, regulation, risk types, etc.\n"
        "\t- Terms should be as atomistic as possible and should be broken down to the most fundamental level of detail."
        "\n\n"

        "Thought 2: Think about how these terms can have relationships with other terms.\n"
        "\t- Terms that are mentioned in the same sentence or paragraph are typically related to each other.\n"
        "\t- Identify specific relationships whenever possible instead of general relationships.\n"
        "\t- The relationships between terms can vary and should include clear connection types, such as:\n"
        "\t\t'is part of', 'includes', 'depends on', 'is related to', 'affects', 'results from', 'is governed by', 'mitigates', 'influences', "
        "'is necessary for', 'supports', 'enables', 'is prerequisite for', 'is defined by', 'is linked to', or 'is managed by'.\n"
        "\t- Avoid defaulting to generic relationships if a specific relationship can be inferred from the context.\n"
        "\t- Relationships can be bidirectional or hierarchical; include relationships such as parent-child structures where applicable. For example, "
        "'Risk Management Framework' contains 'Risk Assessment', which further includes 'Risk Identification' and 'Risk Analysis'.\n\n"

        "Thought 3: When specifying relationships, always prefer the most descriptive and explicit type of relationship that the context suggests.\n"
        "\t- Only use 'contextual_proximity' if no other specific relationship can be accurately identified.\n"
        "\t- Strive to capture the exact nature of how one term relates to another. Be precise in identifying connections such as 'governs', 'mitigates', "
        "'enables', 'depends upon', etc.\n"
        "\t- Provide examples of specific relationships where possible. For instance, if the text mentions 'ISO 31000 helps to standardize risk management processes,' "
        "extract this as 'ISO 31000' (node_1) with relation 'standardizes' (edge) to 'risk management processes' (node_2).\n\n"

        "Format your output as a list of JSON objects. Each element of the list should contain a pair of terms "
        "and the relation between them, like the following: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from the extracted ontology",\n'
        '       "node_2": "A related concept from the extracted ontology",\n'
        '       "edge": "relationship between node_1 and node_2, e.g., is part of, affects, depends on, mitigates, results from"\n'
        "   },\n"
        "   {...}\n"
        "]\n"
        
        "Make sure to:\n"
        "\t1. Represent hierarchical relationships explicitly, if they exist (e.g., 'is part of', 'contains').\n"
        "\t2. Avoid defaulting to 'contextual_proximity' unless absolutely necessary.\n"
        "\t3. Include as many one-to-one, clear and specific relationships as possible that help in understanding the interconnections between risk management concepts.\n"
        "\t4. Use examples to help in inferring relationships wherever there is ambiguity; for instance, use 'is governed by', 'is prerequisite for', "
        "or 'is influenced by' if the context clearly implies such a relationship.\n"
    )



    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result
