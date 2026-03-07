def get_prompts():
    strategies = {
        "chain_of_thought": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy.
    You will read the following clinical snippet and THINK STEP BY STEP about whether it indicates cannabis use from the patient.
    First, explain your reasoning in one sentence. Then classify the snippet as:
    1 = Not A True Mention
    2 = Denial of Use
    3 = Positive Past Use
    4 = Positive Current Use

    Snippet:
    {text}

    After your reasoning, respond with a JSON object in this exact format:
    {{"classification": <number between 1-4>}}
    """.strip(),

        "zero_shot": lambda text: f"""
    Read the snippet and classify whether this snippet is:
    1 = Not A True Mention
    2 = Denial of Use
    3 = Positive Past Use
    4 = Positive Current Use

    Snippet:
    {text}

    Respond ONLY with a JSON object in this exact format:
    {{"classification": <number between 1-4>}}
    """.strip(),

        "one_shot": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy.
    Here are examples for your reference:
    Example 1: " "
    Classification: 1 (Not A True Mention)

    Example 2: " "
    Classification: 2 (Denial of Use)

    Example 3: " "
    Classification: 3 (Positive Past Use)

    Example 4: " "
    Classification: 4 (Positive Current Use)

    Now classify this snippet:
    {text}

    Respond with a JSON object in this exact format:
    {{"classification": <number between 1-4>}}
    """.strip(),

            "two_shot": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy.
    Here are examples for your reference:
    Example: " "
    Classification: 1 (Not A True Mention)

    Example: " "
    Classification: 1 (Not A True Mention)

    Example: " "
    Classification: 2 (Denial of Use)

    Example: " "
    Classification: 2 (Denial of Use)

    Example: " "
    Classification: 3 (Positive Past Use)

    Example: " "
    Classification: 3 (Positive Past Use)

    Example: " "
    Classification: 4 (Positive Current Use)

    Example: " "
    Classification: 4 (Positive Current Use)

    Now classify this snippet:
    {text}

    Respond with a JSON object in this exact format:
    {{"classification": <number between 1-4>}}
    """.strip(),


    "structured_reasoning": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy using the following framework:

    ANALYSIS:
    1. Context identification: [Is this about the patient, family history, or general discussion?]
    2. Temporal markers: [Look for words indicating past, present, or future]
    3. Negation detection: [Check for denial, "no", "denies", etc.]
    4. Certainty assessment: [Is the statement definitive or uncertain?]

    CLASSIFICATION:
    1 = Not A True Mention
    2 = Denial of Use
    3 = Positive Past Use
    4 = Positive Current Use

    SNIPPET: {text}

    After your analysis, respond with a JSON object in this exact format:
    {{"classification": <number between 1-4>}}
    """.strip(),

        "role_specific": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy.

    Review this snippet as you would in your professional practice:
    {text}

    Based on your expertise, classify this mention:
    1 = Not A True Mention
    2 = Denial of Use
    3 = Positive Past Use
    4 = Positive Current Use

    Respond with a JSON object in this exact format:
    {{"classification": <number between 1-4>}}
    """.strip(),

        "confidence_scoring": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy.
    Classify this clinical snippet for cannabis use and provide your confidence level:

    Classification:
    1 = Not A True Mention
    2 = Denial of Use
    3 = Positive Past Use
    4 = Positive Current Use

    Snippet: {text}

    First provide your confidence level and reasoning, then respond with a JSON object containing your classification:
    {{"classification": <number between 1-4>, "confidence": "<Low/Medium/High>", "reasoning": "<brief explanation>"}}
    """.strip(),

        "multi_step_verification": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy using the following verification steps:
    Step 1: Does this snippet mention cannabis/marijuana/THC or related terms? (Yes/No)

    Step 2: If yes, is this referring to the patient specifically? (Yes/No/Unclear)

    Step 3: What is the temporal context?
    1 = Not A True Mention
    2 = Denial of Use
    3 = Positive Past Use
    4 = Positive Current Use

    Step 4: Final classification for snippet: "{text}"

    After your reasoning, respond with a JSON object in this exact format:
    {{"classification": <number between 1-4>}}
    """.strip(),

        "uncertainty_handling": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify cannabis use from clinical notes with high accuracy.
    Ambiguity is common in clinical notes. If you encounter uncertainty, use classification 1.

    Snippet: {text}

    Analysis:
    - Clear indicators present: [Yes/No]
    - Ambiguous elements: [List any unclear aspects]
    - Missing context: [What information would help?]

    If certain, provide classification 1-4:
    1 = Not A True Mention
    2 = Denial of Use
    3 = Positive Past Use
    4 = Positive Current Use

    After your analysis, respond with a JSON object in this exact format:
    {{"classification": <number between 1-4>, "certainty": "<High/Medium/Low>"}}
    """.strip()
    }

    return strategies
